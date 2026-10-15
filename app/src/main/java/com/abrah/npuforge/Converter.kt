package com.abrah.npuforge

import android.annotation.SuppressLint
import android.app.ActivityManager
import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import java.io.BufferedReader
import java.io.File
import java.io.FileOutputStream
import java.io.InputStreamReader
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.runInterruptible
import kotlinx.coroutines.withContext
import java.util.zip.Deflater
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import kotlin.time.Duration.Companion.seconds

/**
 * The conversion pipeline: a `.safetensors` in, a model directory out.
 *
 * QNN graphs are populated by [stageWeights], then compiled by [stageCompile].
 * Both model families write checkpoint-owned MNN text encoders via [stageClip].
 */
object Converter {

    private const val TAG = "Converter"

    class Failure(message: String, val log: String = "") : Exception(message)

    const val OUTPUT_SUBDIR = "npuforge"
    const val SWAP_MARKER = "lora_targets.json"
    const val IP_MARKER = "ip_targets.json"
    const val SWAP_FEATURES_MARKER = "swap_features.json"
    const val V_PRED_MARKER = "V_PRED"

    internal fun predictionMarkers(type: CheckpointInfo.PredictionType): Map<String, String> =
        if (type == CheckpointInfo.PredictionType.V_PREDICTION) mapOf(V_PRED_MARKER to "")
        else emptyMap()

    val SWAP_FEATURES = listOf("lora", "cn", "ip", "inp", "conv", "freeu", "pag", "couple")

    private val UNOFFERED = mapOf(
        CheckpointInfo.Model.SDXL_SWAP to setOf("inp", "conv", "freeu", "pag", "couple"),
    )

    fun sdxlSwapTemplate(context: Context): Pair<String, Int> = runCatching {
        val text = context.assets.open("${CheckpointInfo.Model.SDXL_SWAP.templateDirectory}/$SWAP_FEATURES_MARKER")
            .use { it.readBytes().toString(Charsets.UTF_8) }
        val name = Regex("\"template\"\\s*:\\s*\"(\\w+)\"").find(text)?.groupValues?.get(1) ?: "sdxl_swap_v1"
        val tokens = Regex("\"text_tokens\"\\s*:\\s*(\\d+)").find(text)?.groupValues?.get(1)?.toInt() ?: 231
        name to tokens
    }.getOrDefault("sdxl_swap_v1" to 231)

    fun swapTemplateFeatures(context: Context, model: CheckpointInfo.Model): List<String> = runCatching {
        val text = context.assets.open("${model.templateDirectory}/$SWAP_FEATURES_MARKER")
            .use { it.readBytes().toString(Charsets.UTF_8) }
        val listed = Regex("\"(\\w+)\"").findAll(text.substringAfter("[")).map { it.groupValues[1] }.toSet()
        SWAP_FEATURES.filter { it in listed }
    }.getOrDefault(emptyList())

    fun swapFeaturesSupported(
        context: Context,
        model: CheckpointInfo.Model = CheckpointInfo.Model.SD15_SWAP,
    ): List<String> = swapTemplateFeatures(context, model) - UNOFFERED[model].orEmpty()

    fun swapFeaturesJson(kept: Collection<String>, template: String = "swap_v3"): String =
        "{\"template\":\"$template\",\"features\":[" +
            SWAP_FEATURES.filter { it in kept }.joinToString(",") { "\"$it\"" } + "]}"

    private fun featureDetail(model: CheckpointInfo.Model, feature: String): String? {
        val xl = model == CheckpointInfo.Model.SDXL_SWAP
        return when (feature) {
            "lora" -> "{\"targets\":\"$SWAP_MARKER\",\"rank\":64}"
            "cn" -> if (xl) "{\"residuals\":10,\"hint\":1024}" else "{\"residuals\":13,\"hint\":512}"
            "ip" -> "{\"targets\":\"$IP_MARKER\",\"layers\":${if (xl) 70 else 16}}"
            "inp" -> "{\"mask\":${if (xl) 128 else 64}}"
            "conv" -> "{\"targets\":\"conv_targets.json\",\"rank\":32}"
            else -> null
        }
    }

    fun manifestJson(
        model: CheckpointInfo.Model,
        kept: Collection<String>,
        template: String,
        textTokens: Int,
        prediction: CheckpointInfo.PredictionType,
        producer: String,
        soc: String,
    ): String {
        val family = when (model) {
            CheckpointInfo.Model.SDXL, CheckpointInfo.Model.SDXL_SWAP -> "sdxl"
            else -> "sd15"
        }
        val kind = when (model) {
            CheckpointInfo.Model.SD15_SWAP, CheckpointInfo.Model.SDXL_SWAP -> "swap"
            CheckpointInfo.Model.SD15_INPAINT -> "inpaint"
            else -> "plain"
        }
        val features = SWAP_FEATURES.filter { it in kept }
        val detail = features.mapNotNull { f -> featureDetail(model, f)?.let { "\"$f\":$it" } }
        fun q(s: String) = "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\""
        return "{\"schema\":2,\"producer\":${q(producer)},\"template\":${q(template)}," +
            "\"family\":\"$family\",\"kind\":\"$kind\"," +
            "\"prediction\":\"${if (prediction == CheckpointInfo.PredictionType.V_PREDICTION) "v" else "eps"}\"," +
            "\"text_tokens\":$textTokens,\"size\":${if (family == "sdxl") 1024 else 512},\"soc\":${q(soc)}," +
            "\"detail\":{${detail.joinToString(",")}}," +
            "\"features\":[" + features.joinToString(",") { "\"$it\"" } + "]}"
    }

    /**
     * Resolves target HTP hardware parameters tailored for Snapdragon processors.
     * Prevents MODEL_GRAPH_ERROR (exit code 14) on 7-series SoCs by clamping VTCM to 4 MB
     * and setting soc_model to 0 (dynamic runtime binding).
     */
    fun resolveHtpHardwareConfig(socName: String): Triple<String, Int, Int> {
        val soc = socName.uppercase()
        val hardware = Build.HARDWARE.uppercase()
        val model = Build.MODEL.uppercase()

        return when {
            // Snapdragon 7 Gen 4 (SM7750) & related series
            soc.contains("SM7750") || hardware.contains("SM7750") || model.contains("CPH2825") -> {
                Triple("v73", 0, 4)
            }
            // Snapdragon 8 Gen 3 (SM8650)
            soc.contains("SM8650") || hardware.contains("SM8650") -> {
                Triple("v75", 57, 8)
            }
            // Snapdragon 8 Gen 2 (SM8550)
            soc.contains("SM8550") || hardware.contains("SM8550") -> {
                Triple("v73", 43, 8)
            }
            // Fallback for general Snapdragon devices: 4 MB VTCM, auto soc_model
            else -> {
                Triple("v73", 0, 4)
            }
        }
    }

    /**
     * Generates the backend JSON configuration for libqnncontextgen.so.
     */
    fun buildHtpConfigJson(socName: String, graphName: String = "model"): String {
        val (dspArch, socModel, vtcmMb) = resolveHtpHardwareConfig(socName)
        return """
        {
          "graphs": [
            {
              "vtcm_mb": $vtcmMb,
              "graph_names": [
                "$graphName"
              ],
              "O": 3.0
            }
          ],
          "devices": [
            {
              "dsp_arch": "$dspArch",
              "soc_model": $socModel,
              "cores": [
                {
                  "perf_profile": "burst",
                  "rpc_control_latency": 100
                }
              ]
            }
          ]
        }
        """.trimIndent()
    }

    /** Copies checkpoint or LoRA data into this conversion's work directory. */
    suspend fun importFile(context: Context, uri: Uri, dst: File) = withContext(Dispatchers.IO) {
        context.contentResolver.openInputStream(uri)?.use { input ->
            FileOutputStream(dst).use { output ->
                input.copyTo(output)
            }
        } ?: throw Failure("Could not open input URI: $uri")
    }

    /**
     * Executes native helper binaries placed in the native library directory.
     */
    suspend fun runNativeCommand(
        command: List<String>,
        cwd: File,
        env: Map<String, String>,
        onLog: (String) -> Unit
    ): Int = withContext(Dispatchers.IO) {
        val processBuilder = ProcessBuilder(command)
            .directory(cwd)
            .redirectErrorStream(true)

        processBuilder.environment().putAll(env)

        val process = processBuilder.start()
        val reader = BufferedReader(InputStreamReader(process.inputStream))

        try {
            var line: String?
            while (reader.readLine().also { line = it } != null) {
                line?.let { onLog(it) }
            }
            process.waitFor()
        } finally {
            reader.close()
            process.destroy()
        }
    }

    /**
     * Executes the conversion pipeline.
     */
    suspend fun convert(
        context: Context,
        checkpointUri: Uri,
        modelInfo: CheckpointInfo.Model,
        predictionType: CheckpointInfo.PredictionType,
        socName: String,
        skipVaeEncoder: Boolean = true,
        onProgress: (String) -> Unit
    ) = withContext(Dispatchers.Default) {
        val workDir = File(context.noBackupFilesDir, "conversion-work").apply {
            deleteRecursively()
            mkdirs()
        }

        try {
            val ckptFile = File(workDir, "ckpt.safetensors")
            onProgress("Importing checkpoint…")
            importFile(context, checkpointUri, ckptFile)

            val nativeDir = context.applicationInfo.nativeLibraryDir
            val qnnLibsDir = File(context.filesDir, "qnnlibs").absolutePath

            val baseEnv = mapOf(
                "LD_LIBRARY_PATH" to "$nativeDir:/system/lib64:/vendor/lib64:/vendor/lib64/egl",
                "ADSP_LIBRARY_PATH" to "$qnnLibsDir;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp"
            )

            // Stage 1: Text Encoders (CLIP)
            onProgress("Converting text encoders…")
            val clipDir = File(workDir, "clip").apply { mkdirs() }
            val clipExit = runNativeCommand(
                command = listOf(
                    "$nativeDir/libcomponentconv.so",
                    clipDir.absolutePath,
                    ckptFile.absolutePath,
                    File(workDir, "components").apply { mkdirs() }.absolutePath
                ),
                cwd = workDir,
                env = emptyMap(),
                onLog = { Log.d(TAG, it) }
            )
            if (clipExit != 0) throw Failure("CLIP conversion failed (rc $clipExit)")

            // Stage 2: VAE Encoder (Optional: bypassed by default for Text-to-Image models)
            if (!skipVaeEncoder) {
                onProgress("Preparing image encoder weights…")
                val vaeDir = File(workDir, "vae_encoder").apply { mkdirs() }
                val outPack = File(vaeDir, "out.pack")
                
                val tplExit = runNativeCommand(
                    command = listOf(
                        "$nativeDir/libtplconv.so",
                        File(vaeDir, "template/recipe.bin").absolutePath,
                        File(vaeDir, "template/tpl_trim.pack").absolutePath,
                        ckptFile.absolutePath,
                        outPack.absolutePath
                    ),
                    cwd = vaeDir,
                    env = emptyMap(),
                    onLog = { Log.d(TAG, it) }
                )
                if (tplExit != 0) throw Failure("libtplconv.so failed (rc $tplExit)")

                onProgress("Building the image encoder…")
                val htpConfigFile = File(vaeDir, "htp_backend.json").apply {
                    writeText(buildHtpConfigJson(socName, "model"))
                }

                val compileEnv = baseEnv.toMutableMap().apply {
                    put("QNN_TPL_PACK", outPack.absolutePath)
                }

                val qnnExit = runNativeCommand(
                    command = listOf(
                        "$nativeDir/libqnncontextgen.so",
                        "--model", File(vaeDir, "template/libqnn_model.so").absolutePath,
                        "--backend", "$nativeDir/libQnnHtp.so",
                        "--output_dir", File(vaeDir, "out").apply { mkdirs() }.absolutePath,
                        "--binary_file", "vae_encoder",
                        "--config_file", htpConfigFile.absolutePath,
                        "--log_level", "info"
                    ),
                    cwd = vaeDir,
                    env = compileEnv,
                    onLog = { Log.d(TAG, it) }
                )
                if (qnnExit != 0) throw Failure("libqnncontextgen.so failed (rc $qnnExit)")
            } else {
                Log.i(TAG, "Skipping VAE Encoder compilation (not required for Text-to-Image generation)")
            }

            onProgress("Conversion finished successfully")
        } finally {
            workDir.deleteRecursively()
        }
    }
}
