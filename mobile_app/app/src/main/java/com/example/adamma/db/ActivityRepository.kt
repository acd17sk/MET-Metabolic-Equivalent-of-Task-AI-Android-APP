package com.example.adamma.db

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.text.SimpleDateFormat
import java.util.*

class ActivityRepository(context: Context) {
    private val dao = AppDatabase.getDatabase(context).activityDao()

    private fun todayDate(): String {
        return SimpleDateFormat("yyyy-MM-dd", Locale.getDefault()).format(Date())
    }

    suspend fun logActivity(metClass: String, durationSec: Int) {
        withContext(Dispatchers.IO) {
            dao.insert(ActivityEntity(date = todayDate(), metClass = metClass, durationSec = durationSec))
        }
    }

    suspend fun getTodayActivities(): List<ActivityEntity> {
        return withContext(Dispatchers.IO) {
            dao.getActivitiesByDate(todayDate())
        }
    }

    suspend fun clearToday() {
        withContext(Dispatchers.IO) {
            dao.clearDay(todayDate())
        }
    }

    suspend fun getAllActivities(): List<ActivityEntity> {
        return withContext(Dispatchers.IO) {
            dao.getAllActivities()
        }
    }
}
