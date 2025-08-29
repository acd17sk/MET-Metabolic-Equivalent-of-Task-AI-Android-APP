package com.example.adamma

import android.content.Context
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

object Scaler {
    lateinit var rawMeans: FloatArray
    lateinit var rawStds: FloatArray
    lateinit var featMeans: FloatArray
    lateinit var featStds: FloatArray

    fun load(context: Context) {
        rawMeans = loadNpy(context, "raw_means.npy")
        rawStds  = loadNpy(context, "raw_stds.npy")
        featMeans = loadNpy(context, "feat_means.npy")
        featStds  = loadNpy(context, "feat_stds.npy")
    }

    fun scaleRaw(window: FloatArray): FloatArray {
        val out = FloatArray(window.size)
        val numCh = rawMeans.size // 8 channels
        for (i in window.indices) {
            val ch = i % numCh
            out[i] = (window[i] - rawMeans[ch]) / rawStds[ch]
        }
        return out
    }

    fun scaleFeat(features: FloatArray): FloatArray {
        val out = FloatArray(features.size)
        for (i in features.indices) {
            out[i] = (features[i] - featMeans[i]) / featStds[i]
        }
        return out
    }

    private fun loadNpy(context: Context, assetName: String): FloatArray {
        val inp = context.assets.open(assetName)
        val bytes = inp.readBytes()
        inp.close()

        // Find the end of the header (look for the first newline after the header section)
        var headerEnd = -1
        for (i in bytes.indices) {
            if (bytes[i].toInt() == 0x0A) { // newline '\n'
                headerEnd = i
                break
            }
        }
        if (headerEnd == -1) {
            throw IllegalArgumentException("Invalid NPY file: no header found in $assetName")
        }

        // The actual float32 data starts after the header newline + 1
        val data = bytes.copyOfRange(headerEnd + 1, bytes.size)

        val bb = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN)
        val fb = bb.asFloatBuffer()
        val arr = FloatArray(fb.remaining())
        fb.get(arr)
        return arr
    }

}
