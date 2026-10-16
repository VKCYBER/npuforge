package com.abrah.npuforge

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.net.Uri
import android.os.IBinder
import android.os.Build
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.content.IntentCompat
import androidx.core.net.toUri
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.time.Duration.Companion.minutes

/**
 * Owns one conversion and its foreground notification until native work and cleanup finish.
 */
class ConvertService : Service() {

    sealed interface State {
        data object Idle : State
        data class Running(
            val stage: String,
            val detail: String = "",
            val step: Int = 0,
            val steps: Int = 0,
            val log: List<String> = emptyList(),
            val startedAt: Long = 0,
            /** What is being converted, one line: `name · SDXL Swap · LoRA · V-pred`. */
            val config: String = "",
        ) : State
        data class Done(val dir: String, val seconds: Long, val log: String = "", val config: String = "") : State
        data class Failed(val message: String, val log: String = "") : State
    }

    companion object {
        private const val TAG = "ConvertService"
        private const val CHANNEL = "convert"
        internal const val NOTE_ID = 1
        /** The finished/failed notice: its own id, so it outlives the foreground note. */
        private const val DONE_ID = 2
        /** Default importance, unlike [CHANNEL]: a finish after an hour must actually be seen. */
        private const val CHANNEL_DONE = "convert_done"
        private const val ACTION_STOP = "com.abrah.npuforge.STOP_CONVERSION"
        const val ACTION_DOWNLOAD_VAE = "com.abrah.npuforge.DOWNLOAD_VAE"
        private val WAKE_LOCK_LEASE = 10.minutes
        private val PERCENT = Regex("""(\d{1,3})%""")
        const val EXTRA_URI = "uri"
        const val EXTRA_NAME = "name"
        const val EXTRA_LORAS = "loras"
        const val EXTRA_MODEL = "model"
        /** Plain 4-channel SD1.5 checkpoint -> inpainting model by add-difference ([InpaintDiff]). */
        const val EXTRA_INPAINT_DIFF = "inpaint_diff"
        /** A user-downloaded copy of the inpainting difference, imported instead of downloading. */
        const val EXTRA_INPAINT_DIFF_URI = "inpaint_diff_uri"
        /** SD1.5 text-encoder clip skip, 1 or 2 ([CheckpointInfo.clipDirectory]). */
        const val EXTRA_CLIP_SKIP = "clip_skip"
        /** Explicit UI/automation override; absent means use trusted checkpoint metadata. */
        const val EXTRA_PREDICTION_TYPE = "prediction_type"
        /**
         * SD1.5 Swap v3: the features to KEEP, comma-separated ([Converter.SWAP_FEATURES]);
         * absent = every feature the template supports. "inp" makes an inpainting model by
         * add-difference. A plain string so `adb shell am --es` can drive it.
         */
        const val EXTRA_SWAP_FEATURES = "swap_features"
        const val EXTRA_SKIP_VAE_ENCODER = "skip_vae_encoder"

        private val _state = MutableStateFlow<State>(State.Idle)
        val state: StateFlow<State> = _state.asStateFlow()

        fun downloadVae(context: Context) {
            val i = Intent(context, ConvertService::class.java).setAction(ACTION_DOWNLOAD_VAE)
            context.startForegroundService(i)
        }

        fun start(
            context: Context,
            uri: Uri,
            name: String,
            loras: List<Pair<Uri, Float>> = emptyList(),
            model: CheckpointInfo.Model = CheckpointInfo.Model.SD15,
            inpaintDiff: Boolean = false,
            inpaintDiffUri: Uri? = null,
            clipSkip: Int = 2,
            swapFeatures: Collection<String>? = null,
            predictionType: CheckpointInfo.PredictionType? = null,
            skipVaeEncoder: Boolean? = null,
        ) {
            val i = Intent(context, ConvertService::class.java)
                .putExtra(EXTRA_URI, uri)
                .putExtra(EXTRA_NAME, name)
                .putExtra(EXTRA_MODEL, model.name)
                .putExtra(EXTRA_INPAINT_DIFF, inpaintDiff)
                .putExtra(EXTRA_INPAINT_DIFF_URI, inpaintDiffUri)
                .putExtra(EXTRA_CLIP_SKIP, clipSkip)
                .putExtra(EXTRA_LORAS, loras.map { "${it.first}\vert{}${it.second}" }.toTypedArray())
            if (swapFeatures != null) i.putExtra(EXTRA_SWAP_FEATURES, swapFeatures.joinToString(","))
            if (predictionType != null) i.putExtra(EXTRA_PREDICTION_TYPE, predictionType.name)
            if (skipVaeEncoder != null) i.putExtra(EXTRA_SKIP_VAE_ENCODER, skipVaeEncoder)
            context.startForegroundService(i)
        }

        fun reset() {
            if (_state.value !is State.Running) _state.value = State.Idle
        }
    }

    private val job: Job = SupervisorJob()
    private val scope = CoroutineScope(Dispatchers.Main.immediate + job)
    private var conversion: Job? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(
            NotificationChannel(CHANNEL, getString(R.string.channel_convert),
                NotificationManager.IMPORTANCE_LOW)
        )
        nm.createNotificationChannel(
            NotificationChannel(CHANNEL_DONE, getString(R.string.channel_convert_done),
                NotificationManager.IMPORTANCE_DEFAULT)
        )
    }

    private fun note(text: String): Notification =
        NotificationCompat.Builder(this, CHANNEL)
            .setContentTitle(
                if (convertName.isEmpty()) getString(R.string.app_name)
                else getString(R.string.converting_title, convertName)
            )
            .setContentText(text)
            .setStyle(
                NotificationCompat.BigTextStyle().bigText(
                    if (config.isEmpty()) text else text + "\n" + config
                )
            )
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
            .setContentIntent(PendingIntent.getActivity(
                this, 0, Intent(this, MainActivity::class.java)
                    .addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP),
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
            ))
            .addAction(0, getString(R.string.stop_conversion), PendingIntent.getService(
                this, 1, Intent(this, ConvertService::class.java).setAction(ACTION_STOP),
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
            ))
            .setProgress(0, 0, true)
            .build()

    private var step = 0
    private var steps = 0
    private var startedAt = 0L
    private var convertName = ""
    private var config = ""
    private val logLines = ArrayDeque<String>()
    private var report: ConversionReport? = null

    private fun post(stage: String, detail: String = "") {
        job.ensureActive()
        if ((_state.value as? State.Running)?.stage != stage) {
            report?.record("stage=$stage")
            logLines.addLast(stage)
        }
        while (logLines.size > 200) logLines.removeFirst()
        _state.value = State.Running(stage, detail, step, steps, logLines.toList(), startedAt, config)
        getSystemService(NotificationManager::class.java)
            .notify(NOTE_ID, note(if (steps > 0) "$step/$steps$stage" else stage))
    }

    private fun logLine(stage: String, line: String) {
        val t = line.trim()
        if (t.isEmpty()) return
        logLines.addLast(t)
        val pct = PERCENT.findAll(t).lastOrNull()?.groupValues?.get(1)
        if (pct != null) {
            post(stage, "$pct%")
            return
        }
        post(stage, t.take(80))
    }

    private suspend fun downloadSdxlVaeFiles(onProgress: (name: String, detail: String) -> Unit) {
        val vaeDir = File(filesDir, "vae_sdxl").apply { mkdirs() }
        for (name in listOf("vae_decoder.bin", "vae_encoder.bin")) {
            val cached = File(vaeDir, name)
            if (cached.isFile && cached.length() > 0L) continue
            onProgress(name, "")
            val part = File(vaeDir, "$name.part")
            var lastPct = -1
            try {
                HfDownload.fetch(
                    this, "Mr-J-369/Fancy-AI/resolve/main/$name", part,
                    onProgress = { done, total ->
                        val pct = (done * 100 / total).toInt()
                        if (pct != lastPct) {
                            lastPct = pct
                            onProgress(name, "$pct%")
                        }
                    },
                    onRetry = { onProgress(name, "retrying ($it)") },
                )
            } catch (e: java.io.IOException) {
                throw Converter.Failure(
                    "SDXL VAE download failed: ${e.message}. Progress is kept; retry, or import a VAE zip from the Utility tab.",
                )
            }
            if (!part.renameTo(cached)) throw Converter.Failure("could not store $name")
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            conversion?.cancel()
            if (conversion == null || conversion?.isCompleted == true) {
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
            }
            return START_NOT_STICKY
        }
        if (intent?.action == ACTION_DOWNLOAD_VAE) {
            convertName = ""
            config = ""
            val stage = getString(R.string.stage_vae_download)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                startForeground(NOTE_ID, note(stage),
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
            } else {
                startForeground(NOTE_ID, note(stage))
            }
            startedAt = SystemClock.elapsedRealtime()
            logLines.clear()
            step = 0
            steps = 1
            post(stage)
            conversion = scope.launch {
                val wakeLock = getSystemService(PowerManager::class.java)
                    .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "npuforge:vae_download")
                    .apply { setReferenceCounted(false) }
                try {
                    wakeLock.acquire(WAKE_LOCK_LEASE.inWholeMilliseconds)
                    withContext(Dispatchers.IO) {
                        downloadSdxlVaeFiles { name, progress ->
                            post(stage, "$name$progress".trim())
                        }
                    }
                    _state.value = State.Idle
                } catch (e: CancellationException) {
                    _state.value = State.Idle
                    throw e
                } catch (e: Exception) {
                    _state.value = State.Failed(e.message ?: e.javaClass.simpleName, logLines.joinToString("\n"))
                } finally {
                    wakeLock.release()
                    stopForeground(STOP_FOREGROUND_REMOVE)
                    stopSelf()
                }
            }
            return START_NOT_STICKY
        }

        if (conversion?.isCompleted == false) {
            Log.i(TAG, "Ignoring duplicate start; conversion is already running")
            return START_NOT_STICKY
        }
        val uri = intent?.let {
            IntentCompat.getParcelableExtra(it, EXTRA_URI, Uri::class.java)
        }
        val name = intent?.getStringExtra(EXTRA_NAME).orEmpty().ifBlank { "converted" }
        val model = CheckpointInfo.Model.valueOf(
            intent?.getStringExtra(EXTRA_MODEL) ?: CheckpointInfo.Model.SD15.name
        )
        val inpaintDiff = intent?.getBooleanExtra(EXTRA_INPAINT_DIFF, false) == true &&
            model == CheckpointInfo.Model.SD15_INPAINT
        val inpaintDiffUri = intent?.let {
            IntentCompat.getParcelableExtra(it, EXTRA_INPAINT_DIFF_URI, Uri::class.java)
        }
        val swapModel = model == CheckpointInfo.Model.SD15_SWAP || model == CheckpointInfo.Model.SDXL_SWAP
        val swapSupported = if (swapModel) Converter.swapFeaturesSupported(this, model) else emptyList()
        val swapKept: List<String>? = if (swapSupported.isEmpty()) null else {
            val asked = intent?.getStringExtra(EXTRA_SWAP_FEATURES)?.split(',')?.map { it.trim() }?.toSet()
            swapSupported.filter { asked == null || it in asked }
        }
        val swapDrop = swapKept?.let { kept ->
            Converter.swapTemplateFeatures(this, model).filterNot { it in kept }.joinToString(",")
        }.orEmpty()
        val swapInpaint = swapKept?.contains("inp") == true
        val clipSkip = if (model.isSdxl) 2
            else intent?.getIntExtra(EXTRA_CLIP_SKIP, 2)?.takeIf { it == 1 } ?: 2
        val requestedPredictionType = intent?.getStringExtra(EXTRA_PREDICTION_TYPE)?.let { value ->
            runCatching { CheckpointInfo.PredictionType.valueOf(value) }.getOrNull()
        }
        val clipDirectory = CheckpointInfo.clipDirectory(model, clipSkip)
        val templateDir = model.templateDirectory
        val loraSpecs: List<Pair<Uri, Float>> =
            (intent?.getStringArrayExtra(EXTRA_LORAS) ?: emptyArray()).mapNotNull { spec ->
                val bar = spec.lastIndexOf('|')
                if (bar <= 0) null
                else spec.substring(0, bar).toUri() to
                    (spec.substring(bar + 1).toFloatOrNull() ?: 1f)
            }
        if (uri == null) {
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
            return START_NOT_STICKY
        }

        val soc = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) Build.SOC_MODEL.uppercase() else ""
        val isMidRangeSoC = soc.contains("SM7750") ||
            Build.HARDWARE.contains("SM7750", ignoreCase = true) ||
            Build.MODEL.contains("CPH2825", ignoreCase = true)[span_1](start_span)[span_1](end_span)
        val skipVaeEncoder = intent?.getBooleanExtra(EXTRA_SKIP_VAE_ENCODER, isMidRangeSoC) ?: isMidRangeSoC

        convertName = name
        config = listOfNotNull(
            name,
            when (model) {
                CheckpointInfo.Model.SD15 -> "SD 1.5"
                CheckpointInfo.Model.SD15_INPAINT ->
                    if (inpaintDiff) "SD 1.5 → " + getString(R.string.mode_inpaint)
                    else "SD 1.5 " + getString(R.string.mode_inpaint)
                CheckpointInfo.Model.SD15_SWAP -> getString(R.string.mode_swap)
                CheckpointInfo.Model.SDXL -> "SDXL"
                CheckpointInfo.Model.SDXL_SWAP -> getString(R.string.mode_sdxl_swap)
            },
            swapKept?.let { kept ->
                if (kept.isEmpty()) getString(R.string.config_no_features)
                else kept.joinToString(" + ") { featureLabel(it) }
            },
            loraSpecs.size.takeIf { it > 0 }?.let { getString(R.string.config_loras_baked, it) },
            if (model.isSdxl) null else getString(R.string.config_clip_skip, clipSkip),
            if (requestedPredictionType == CheckpointInfo.PredictionType.V_PREDICTION) "V-pred" else null,
            if (skipVaeEncoder && !model.isSdxl) "No-Enc" else null,
        ).joinToString(" · ")
        getSystemService(NotificationManager::class.java).cancel(DONE_ID)
        val firstStage = getString(R.string.stage_import)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(NOTE_ID, note(firstStage),
                ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
        } else {
            startForeground(NOTE_ID, note(firstStage))
        }

        startedAt = SystemClock.elapsedRealtime()
        logLines.clear()
        step = 0
        steps = 0
        post(firstStage)
        conversion = scope.launch {
            var result: State = State.Idle
            val work = File(noBackupFilesDir, "conversion-work")
            val wakeLock = getSystemService(PowerManager::class.java)
                .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "npuforge:conversion")
                .apply { setReferenceCounted(false) }
            var wakeLockRenewal: Job? = null
            try {
                wakeLock.acquire(WAKE_LOCK_LEASE.inWholeMilliseconds)
                wakeLockRenewal = launch {
                    while (true) {
                        delay(WAKE_LOCK_LEASE / 2)
                        wakeLock.acquire(WAKE_LOCK_LEASE.inWholeMilliseconds)
                    }
                }
                withContext(Dispatchers.IO) {
                    val diagnostic = ConversionReport(this@ConvertService, "$name (${model.name})")
                    report = diagnostic
                    diagnostic.record("LoRA strengths=${loraSpecs.map { it.second }}")
                    diagnostic.record("Inpaint add-difference=$inpaintDiff")
                    diagnostic.record("Clip skip=$clipSkip")
                    diagnostic.record("Skip VAE Encoder=$skipVaeEncoder")
                    if (swapModel) {
                        diagnostic.record("Swap features supported=$swapSupported kept=$swapKept drop=[$swapDrop]")
                    }
                    work.deleteRecursively()
                    work.mkdirs()
                    diagnostic.record("Conversion workspace: ${work.absolutePath}")

                    val vaeSteps = if (model.isSdxl) 0 else if (skipVaeEncoder) 2 else 4
                    steps = (if (model.isSdxl) 6 else 6 + vaeSteps) + loraSpecs.size +
                        (if (inpaintDiff || swapInpaint) 1 else 0)
                    step = 0

                    step++
                    post(getString(R.string.stage_import))
                    val ckpt = Converter.importFile(this@ConvertService, uri, File(work, "ckpt.safetensors")) { bytes ->
                        post(getString(R.string.stage_import), "${bytes / 1_000_000} MB")
                    }

                    step++
                    post(getString(R.string.stage_validate))
                    if (model.isSdxl && Converter.socTier() == "8gen1") {
                        throw Converter.Failure(
                            "SDXL needs Snapdragon 8 Gen 3 or newer. This phone (8 Gen 1) can convert SD1.5 models.",
                        )
                    }
                    if (model == CheckpointInfo.Model.SD15_SWAP && loraSpecs.isNotEmpty()) {
                        throw Converter.Failure(
                            "SD1.5 Swap does not bake LoRAs in; choose them per render in the generating app.",
                        )
                    }
                    if (model == CheckpointInfo.Model.SDXL_SWAP && loraSpecs.isNotEmpty()) {
                        throw Converter.Failure(
                            "SDXL Swap does not bake LoRAs in; they will be chosen per render in the generating app.",
                        )
                    }
                    val checkpointReport = CheckpointInfo.validate(
                        this@ConvertService, ckpt,
                        when {
                            inpaintDiff || model == CheckpointInfo.Model.SD15_SWAP -> CheckpointInfo.Model.SD15
                            model == CheckpointInfo.Model.SDXL_SWAP -> CheckpointInfo.Model.SDXL
                            else -> model
                        },
                        clipSkip,
                    )
                    val predictionType = requestedPredictionType ?: checkpointReport.predictionType
                        ?: CheckpointInfo.PredictionType.EPSILON
                    diagnostic.record(
                        "Prediction type=$predictionType " +
                            if (requestedPredictionType != null) "(explicit)"
                            else if (checkpointReport.predictionType != null) "(checkpoint metadata)"
                            else "(default; metadata absent or unrecognised)",
                    )

                    val loraFiles = loraSpecs.mapIndexed { idx, (u, strength) ->
                        step++
                        post(getString(R.string.stage_lora))
                        Converter.importFile(this@ConvertService, u, File(work, "lora_$idx.safetensors")) { bytes ->
                            post(getString(R.string.stage_lora), "${bytes / 1_000_000} MB")
                        } to strength
                    }

                    val componentFiles = File(work, "components").apply { mkdirs() }

                    step++
                    post(getString(R.string.stage_clip))
                    Converter.stageClip(
                        this@ConvertService, ckpt, work, componentFiles, model, diagnostic, clipDirectory,
                    ) {
                        logLine(getString(R.string.stage_clip), it)
                    }

                    if (model.isSdxl) {
                        downloadSdxlVaeFiles { vaeName, progress ->
                            post(getString(R.string.stage_vae_download), "$vaeName$progress".trim())
                        }
                        val vaeDir = File(filesDir, "vae_sdxl")
                        for (vaeName in listOf("vae_decoder.bin", "vae_encoder.bin")) {
                            File(vaeDir, vaeName).copyTo(File(componentFiles, vaeName), overwrite = true)
                        }
                    } else {
                        val stages = mutableListOf<Triple<String, Int, Int>>()
                        if (!skipVaeEncoder) {
                            stages.add(Triple("vae_encoder", R.string.stage_vae_encoder_weights, R.string.stage_vae_encoder_compile))
                        }
                        stages.add(Triple("vae_decoder", R.string.stage_vae_decoder_weights, R.string.stage_vae_decoder_compile))

                        for ((component, weightsStage, compileStage) in stages) {
                            val componentWork = File(work, component).apply { mkdirs() }
                            step++
                            post(getString(weightsStage))
                            val componentPack = Converter.stageWeights(
                                this@ConvertService, ckpt, componentWork, model, diagnostic,
                                templateDirectory = "${model.componentDirectory}/$component",
                            ) { logLine(getString(weightsStage), it) }
                            step++
                            post(getString(compileStage))
                            val output = Converter.stageCompile(
                                this@ConvertService, componentPack, componentWork, model, diagnostic, component,
                            ) { logLine(getString(compileStage), it) }
                            if (!output.renameTo(File(componentFiles, "$component.bin"))) {
                                throw Converter.Failure("could not retain $component.bin")
                            }
                            componentWork.deleteRecursively()
                        }
                    }

                    val unetArgs = if (inpaintDiff || swapInpaint) {
                        step++
                        val diffFile = InpaintDiff.forModel(model)
                        val diff = if (inpaintDiffUri != null && !diffFile.isReady(this@ConvertService)) {
                            post(getString(R.string.stage_inpaint_diff_import))
                            diffFile.import(this@ConvertService, inpaintDiffUri) {
                                post(getString(R.string.stage_inpaint_diff_import), it)
                            }
                        } else {
                            post(getString(R.string.stage_inpaint_diff_download))
                            diffFile.ensure(this@ConvertService) {
                                post(getString(R.string.stage_inpaint_diff_download), it)
                            }
                        }
                        diagnostic.record("Inpaint difference: ${diff.name}${diff.length()} bytes")
                        listOf("--inpaint-diff", diff.absolutePath) +
                            (if (swapInpaint) listOf("--input-channel-prefix") else emptyList())
                    } else emptyList()

                    step++
                    post(getString(R.string.stage_weights))
                    val pack = Converter.stageWeights(
                        this@ConvertService, ckpt, work, model, diagnostic, loraFiles,
                        templateDirectory = templateDir,
                        extraArgs = unetArgs,
                    ) {
                        logLine(getString(R.string.stage_weights), it)
                    }

                    ckpt.delete()
                    loraFiles.forEach { (file, _) -> file.delete() }
                    diagnostic.snapshot()

                    step++
                    post(getString(R.string.stage_compile))
                    val unet = Converter.stageCompile(
                        this@ConvertService, pack, work, model, diagnostic, tplDrop = swapDrop,
                    ) {
                        logLine(getString(R.string.stage_compile), it)
                    }
                    pack.delete()

                    step++
                    post(getString(R.string.stage_assemble))
                    val where = Converter.assemble(
                        this@ConvertService, unet, name, model, componentFiles, swapFeatures = swapKept,
                        predictionType = predictionType,
                    ) { f ->
                        post(getString(R.string.stage_assemble), f)
                    }

                    result = State.Done(
                        where, (SystemClock.elapsedRealtime() - startedAt) / 1000,
                        logLines.joinToString("\n"),
                    )
                }
            } catch (e: CancellationException) {
                report?.record("Conversion cancelled")
                throw e
            } catch (e: Converter.Failure) {
                report?.record(e.stackTraceToString())
                Log.e(TAG, "conversion failed: ${e.message}\n${e.log}")
                result = State.Failed(e.message ?: "failed", e.log)
            } catch (e: Exception) {
                report?.record(e.stackTraceToString())
                Log.e(TAG, "conversion failed", e)
                result = State.Failed(e.message ?: e.javaClass.simpleName, logLines.joinToString("\n"))
            } finally {
                try {
                    withContext(NonCancellable + Dispatchers.IO) {
                        if (!work.deleteRecursively()) Log.w(TAG, "Could not remove conversion work files")
                        report?.record("Conversion finished: ${result.javaClass.simpleName}")
                        report?.close()
                        report = null
                    }
                } finally {
                    wakeLockRenewal?.cancel()
                    if (wakeLock.isHeld) wakeLock.release()
                    stopForeground(STOP_FOREGROUND_REMOVE)
                    stopSelf()
                    _state.value = (result as? State.Done)?.copy(config = config) ?: result
                    notifyFinished(result)
                }
            }
        }
        return START_NOT_STICKY
    }

    private fun featureLabel(id: String): String = when (id) {
        "lora" -> getString(R.string.swap_feature_lora)
        "cn" -> getString(R.string.swap_feature_cn)
        "ip" -> getString(R.string.swap_feature_ip)
        "inp" -> getString(R.string.swap_feature_inp)
        else -> id
    }

    private fun notifyFinished(result: State) {
        val (title, text) = when (result) {
            is State.Done -> getString(R.string.notify_done_title) to
                getString(R.string.done_title, result.seconds / 60, result.seconds % 60)
            is State.Failed -> getString(R.string.notify_failed_title) to result.message
            else -> return
        }
        val body = listOf(text, config).filter { it.isNotEmpty() }.joinToString("\n")
        val n = NotificationCompat.Builder(this, CHANNEL_DONE)
            .setContentTitle(title)
            .setContentText(convertName.ifEmpty { text })
            .setStyle(NotificationCompat.BigTextStyle().bigText(body))
            .setSmallIcon(
                if (result is State.Done) android.R.drawable.stat_sys_download_done
                else android.R.drawable.stat_notify_error
            )
            .setAutoCancel(true)
            .setCategory(NotificationCompat.CATEGORY_STATUS)
            .setContentIntent(PendingIntent.getActivity(
                this, 2, Intent(this, MainActivity::class.java)
                    .addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP),
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
            ))
            .build()
        runCatching { getSystemService(NotificationManager::class.java).notify(DONE_ID, n) }
    }

    override fun onDestroy() {
        scope.cancel()
        stopForeground(STOP_FOREGROUND_REMOVE)
        super.onDestroy()
    }
}
