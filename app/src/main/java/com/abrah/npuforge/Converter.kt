package com.abrah.npuforge

import android.content.Context
import android.net.Uri
import android.os.Build
import java.io.BufferedReader
import java.io.File
import java.io.FileOutputStream
import java.io.InputStreamReader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * The conversion pipeline: a `.safetensors` in, a model directory out.
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

    fun socTier(): String {
        val hardware = Build.HARDWARE.lowercase()
        val soc = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) Build.SOC_MODEL.lowercase() else ""
        return when {
            soc.contains("sm8650") || hardware.contains("sm8650") -> "8gen3"
            soc.contains("sm8550") || hardware.contains("sm8550") -> "8gen2"
            soc.contains("sm8450") || hardware.contains("sm8450") || soc.contains("sm8475") -> "8gen1"
            soc.contains("sm7750") || hardware.contains("sm7750") -> "7gen4"
            else -> "generic"
        }
    }

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

    fun resolveHtpHardwareConfig(): Triple<String, Int, Int> {
        val soc = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) Build.SOC_MODEL.uppercase() else ""
        val hardware = Build.HARDWARE.uppercase()
        val model = Build.MODEL.uppercase()

        return when {
            soc.contains("SM7750") || hardware.contains("SM7750") || model.contains("CPH2825") -> {
                Triple("v73", 0, 4)
            }
            soc.contains("SM8650") || hardware.contains("SM8650") -> {
                Triple("v75", 57, 8)
            }
            soc.contains("SM8550") || hardware.contains("SM8550") -> {
                Triple("v73", 43, 8)
            }
            else -> {
                Triple("v73", 0, 4)
            }
        }
    }

    fun buildHtpConfigJson(graphName: String = "model"): String {
        val (dspArch, socModel, vtcmMb) = resolveHtpHardwareConfig()
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

    suspend fun importFile(
        context: Context,
        uri: Uri,
        dst: File,
        onProgress: ((Long) -> Unit)? = null,
    ): File = withContext(Dispatchers.IO) {
        context.contentResolver.openInputStream(uri)?.use { input ->
            FileOutputStream(dst).use { output ->
                val buffer = ByteArray(65536)
                var copied = 0L
                var read: Int
                while (input.read(buffer).also { read = it } >= 0) {
                    output.write(buffer, 0, read)
                    copied += read
                    onProgress?.invoke(copied)
                }
            }
        } ?: throw Failure("Could not open input URI: $uri")
        dst
    }

    fun stageDspLibs(context: Context, destination: File = File(context.filesDir, "qnnlibs")): File {
        destination.mkdirs()
        runCatching {
            context.assets.list("qnnlibs")?.forEach { libName ->
                val outFile = File(destination, libName)
                if (!outFile.exists() || outFile.length() == 0L) {
                    context.assets.open("qnnlibs/$libName").use { input ->
                        outFile.outputStream().use { output ->
                            input.copyTo(output)
                        }
                    }
                }
            }
        }
        return destination
    }

    suspend fun stageClip(
        context: Context,
        ckpt: File,
        workDir: File,
        componentFiles: File,
        model: CheckpointInfo.Model,
        diagnostic: ConversionReport,
        clipDirectory: String,
        onLog: (String) -> Unit
    ) = withContext(Dispatchers.IO) {
        val nativeDir = context.applicationInfo.nativeLibraryDir
        val clipDir = File(workDir, "clip").apply { mkdirs() }
        val exit = runNativeCommand(
            command = listOf(
                "$nativeDir/libcomponentconv.so",
                clipDir.absolutePath,
                ckpt.absolutePath,
                componentFiles.absolutePath
            ),
            cwd = workDir,
            env = emptyMap(),
            onLog = onLog
        )
        if (exit != 0) throw Failure("libcomponentconv.so failed (rc $exit)")
    }

    suspend fun stageWeights(
        context: Context,
        ckpt: File,
        workDir: File,
        model: CheckpointInfo.Model,
        diagnostic: ConversionReport,
        loraFiles: List<Pair<File, Float>> = emptyList(),
        templateDirectory: String = model.templateDirectory,
        extraArgs: List<String> = emptyList(),
        onLog: (String) -> Unit
    ): File = withContext(Dispatchers.IO) {
        val nativeDir = context.applicationInfo.nativeLibraryDir
        val outPack = File(workDir, "out.pack")
        val recipe = File(workDir, "recipe.bin")
        val tplTrim = File(workDir, "tpl_trim.pack")

        val command = mutableListOf(
            "$nativeDir/libtplconv.so",
            recipe.absolutePath,
            tplTrim.absolutePath,
            ckpt.absolutePath,
            outPack.absolutePath
        ).apply { addAll(extraArgs) }

        val exit = runNativeCommand(
            command = command,
            cwd = workDir,
            env = emptyMap(),
            onLog = onLog
        )
        if (exit != 0) throw Failure("libtplconv.so failed (rc $exit)")
        outPack
    }

    suspend fun stageCompile(
        context: Context,
        pack: File,
        workDir: File,
        model: CheckpointInfo.Model,
        diagnostic: ConversionReport,
        component: String = "unet",
        tplDrop: String = "",
        onLog: (String) -> Unit
    ): File = withContext(Dispatchers.IO) {
        val nativeDir = context.applicationInfo.nativeLibraryDir
        val qnnLibsDir = stageDspLibs(context).absolutePath
        val htpConfig = File(workDir, "htp_backend.json").apply {
            writeText(buildHtpConfigJson("model"))
        }

        val outDir = File(workDir, "out").apply { mkdirs() }
        val env = mutableMapOf(
            "LD_LIBRARY_PATH" to "$nativeDir:/system/lib64:/vendor/lib64:/vendor/lib64/egl",
            "ADSP_LIBRARY_PATH" to "$qnnLibsDir;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp",
            "QNN_TPL_PACK" to pack.absolutePath
        )
        if (tplDrop.isNotEmpty()) {
            env["QNN_TPL_DROP"] = tplDrop
        }

        val exit = runNativeCommand(
            command = listOf(
                "$nativeDir/libqnncontextgen.so",
                "--model", File(workDir, "template/libqnn_model.so").absolutePath,
                "--backend", "$nativeDir/libQnnHtp.so",
                "--output_dir", outDir.absolutePath,
                "--binary_file", component,
                "--config_file", htpConfig.absolutePath,
                "--log_level", "info"
            ),
            cwd = workDir,
            env = env,
            onLog = onLog
        )
        if (exit != 0) throw Failure("libqnncontextgen.so failed (rc $exit)")
        File(outDir, "$component.bin")
    }

    suspend fun assemble(
        context: Context,
        unet: File,
        name: String,
        model: CheckpointInfo.Model,
        componentFiles: File,
        swapFeatures: Collection<String>? = null,
        predictionType: CheckpointInfo.PredictionType = CheckpointInfo.PredictionType.EPSILON,
        onProgress: (String) -> Unit
    ): String = withContext(Dispatchers.IO) {
        onProgress("Assembling model archive…")
        val exportDir = File(context.filesDir, OUTPUT_SUBDIR).apply { mkdirs() }
        val finalZip = File(exportDir, "$name.zip")
        finalZip.absolutePath
    }

    private suspend fun runNativeCommand(
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
}
