package com.example.adamma

import kotlin.math.sqrt

object FeatureExtractor {
    fun extract(window: List<FloatArray>): FloatArray {
        val n = window.size
        val ax = FloatArray(n); val ay = FloatArray(n); val az = FloatArray(n)
        val gx = FloatArray(n); val gy = FloatArray(n); val gz = FloatArray(n)
        val amag = FloatArray(n); val gmag = FloatArray(n)

        for (i in 0 until n) {
            val r = window[i]
            ax[i] = r[0]; ay[i] = r[1]; az[i] = r[2]
            gx[i] = r[3]; gy[i] = r[4]; gz[i] = r[5]
            amag[i] = sqrt(ax[i]*ax[i] + ay[i]*ay[i] + az[i]*az[i])
            gmag[i] = sqrt(gx[i]*gx[i] + gy[i]*gy[i] + gz[i]*gz[i])
        }

        val feats = ArrayList<Float>(36)

        // helper to compute mean/std/min/max
        fun stats(arr: FloatArray) {
            var s = 0.0; var ss = 0.0
            var mn = Float.POSITIVE_INFINITY; var mx = Float.NEGATIVE_INFINITY
            val nf = arr.size.toFloat()
            for (v in arr) {
                val dv = v.toDouble()
                s += dv; ss += dv*dv
                if (v < mn) mn = v
                if (v > mx) mx = v
            }
            val mean = (s/nf).toFloat()
            val varr = kotlin.math.max(0.0, ss/nf - (s/nf)*(s/nf)).toFloat()
            val std  = sqrt(varr.toDouble()).toFloat()
            feats += mean; feats += std; feats += mn; feats += mx
        }

        // 8 channels × 4 = 32
        stats(ax); stats(ay); stats(az)
        stats(gx); stats(gy); stats(gz)
        stats(amag); stats(gmag)

        // jerk(|acc|) ~ derivative of |acc|
        val jerk = FloatArray(n)
        if (n > 0) jerk[0] = 0f
        for (i in 1 until n) {
            jerk[i] = (amag[i] - amag[i-1]) * 50f  // 50 Hz sampling
        }
        stats(jerk) // +4

        return feats.toFloatArray() // length = 36
    }
}
