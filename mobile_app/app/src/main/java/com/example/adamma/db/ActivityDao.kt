package com.example.adamma.db

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query

@Dao
interface ActivityDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(activity: ActivityEntity)

    @Query("SELECT * FROM activity_log WHERE date = :date")
    suspend fun getActivitiesByDate(date: String): List<ActivityEntity>

    @Query("DELETE FROM activity_log WHERE date = :date")
    suspend fun clearDay(date: String)

    @Query("SELECT * FROM activity_log ORDER BY date DESC")
    suspend fun getAllActivities(): List<ActivityEntity>
}
