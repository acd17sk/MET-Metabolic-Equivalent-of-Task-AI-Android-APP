package com.example.adamma

import ai.onnxruntime.*
import android.content.Context
import android.util.Log
import java.nio.FloatBuffer
import java.util.ArrayDeque
import java.util.Locale

object ModelInference {
    private var session: OrtSession? = null
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()

    // Keep a rolling history of probabilities for smoothing
    private val history = ArrayDeque<FloatArray>()
    private const val HISTORY_SIZE = 3

    fun init(context: Context) {
        if (session == null) {
            try {
                val modelBytes = context.assets.open("hybrid_met.onnx").readBytes()
                session = env.createSession(modelBytes, OrtSession.SessionOptions())
                Log.d("ModelInference", "ONNX hybrid model loaded successfully")
            } catch (e: Exception) {
                Log.e("ModelInference", "Failed to load model", e)
            }
        }
    }

    fun predict(rawScaled: FloatArray, winSize: Int, featScaled: FloatArray): String {
        val localSession = session ?: return "Unknown"

        return try {
            // Raw input [1, winSize, 8]
            val rawShape = longArrayOf(1, winSize.toLong(), 8)
            val rawTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(rawScaled), rawShape)

            // Feature input [1,36]
            val featShape = longArrayOf(1, featScaled.size.toLong())
            val featTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(featScaled), featShape)

            val inputs = mapOf(
                "raw_input" to rawTensor,
                "feat_input" to featTensor
            )

            val results = localSession.run(inputs)
            val rawOutput = results[0].value

            val probs = when (rawOutput) {
                is Array<*> -> (rawOutput[0] as? FloatArray) ?: FloatArray(0)
                is FloatArray -> rawOutput
                else -> FloatArray(0)
            }

            // Keep history for smoothing
            if (probs.isNotEmpty()) {
                history.addLast(probs.clone())
                if (history.size > HISTORY_SIZE) history.removeFirst()
            }

            // Weighted rolling average (newer has higher weight)
            val avgProbs = FloatArray(probs.size)
            val weights = FloatArray(history.size) { i -> (i + 1).toFloat() }
            val totalWeight = weights.sum()

            var j = 0
            for (vec in history) {
                val w = weights[j++]
                for (i in vec.indices) {
                    avgProbs[i] += vec[i] * w
                }
            }
            for (i in avgProbs.indices) {
                avgProbs[i] /= totalWeight
            }

            // Debug: log smoothed probabilities
            if (avgProbs.isNotEmpty()) {
                val formatted = avgProbs.joinToString(
                    prefix = "[", postfix = "]"
                ) { String.format(Locale.US, "%.3f", it) }
                Log.d("ModelInference", "Smoothed probs: $formatted")
            }

            val predicted = avgProbs.indices.maxByOrNull { avgProbs[it] } ?: -1

            return when (predicted) {
                0 -> "Sedentary"
                1 -> "Light"
                2 -> "Moderate"
                3 -> "Vigorous"
                else -> "Unknown"
            }
        } catch (e: Exception) {
            Log.e("ModelInference", "Prediction failed", e)
            "Unknown"
        }
    }
}
