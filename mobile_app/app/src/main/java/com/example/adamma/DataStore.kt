package com.example.adamma

import android.content.Context
import androidx.lifecycle.MutableLiveData
import com.example.adamma.db.ActivityRepository
import com.github.mikephil.charting.data.PieData
import com.github.mikephil.charting.data.PieDataSet
import com.github.mikephil.charting.data.PieEntry
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import com.github.mikephil.charting.formatter.ValueFormatter

class DataStore private constructor(context: Context) {
    val metClassLive = MutableLiveData<String>()

    // Store SECONDS per class
    private val seconds = IntArray(4)
    private val repo = ActivityRepository(context.applicationContext)

    fun resetData() {
        for (i in seconds.indices) {
            seconds[i] = 0
        }
        metClassLive.postValue("Reset")
    }


    fun updateClass(metClass: String, durationSec: Int) {
        val idx = when (metClass) {
            "Sedentary" -> 0
            "Light"     -> 1
            "Moderate"  -> 2
            "Vigorous"  -> 3
            else        -> 0
        }

        // Add actual measured duration
        seconds[idx] += durationSec

        // Notify UI
        metClassLive.postValue(metClass)

        // Persist to DB
        CoroutineScope(Dispatchers.IO).launch {
            repo.logActivity(metClass, durationSec)
        }
    }

    // Custom formatter: show Xm Ys
    class MinuteSecondFormatter : ValueFormatter() {
        override fun getFormattedValue(value: Float): String {
            val totalSeconds = (value * 60).toInt() // value is minutes, convert back to seconds
            val minutes = totalSeconds / 60
            val seconds = totalSeconds % 60
            return if (minutes > 0) {
                "${minutes}m ${seconds}s"
            } else {
                "${seconds}s"
            }
        }
    }

    fun getPieData(): PieData {
        val entries = listOf(
            PieEntry(seconds[0] / 60f, "Sedentary"),
            PieEntry(seconds[1] / 60f, "Light"),
            PieEntry(seconds[2] / 60f, "Moderate"),
            PieEntry(seconds[3] / 60f, "Vigorous")
        )
        val dataSet = PieDataSet(entries, "Daily METs").apply {
            colors = listOf(
                android.graphics.Color.GRAY,
                android.graphics.Color.GREEN,
                android.graphics.Color.CYAN,
                android.graphics.Color.RED
            )
            valueTextSize = 14f
            valueFormatter = MinuteSecondFormatter()
        }
        return PieData(dataSet)
    }

    companion object {
        @Volatile private var INSTANCE: DataStore? = null
        fun getInstance(context: Context): DataStore =
            INSTANCE ?: synchronized(this) {
                INSTANCE ?: DataStore(context).also { INSTANCE = it }
            }
    }
}
