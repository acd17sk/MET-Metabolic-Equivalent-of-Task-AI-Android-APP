package com.example.adamma.db

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "activity_log")
data class ActivityEntity(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val date: String,
    val metClass: String,   // ✅ changed name to match repository
    val durationSec: Int    // ✅ changed name to match repository
)
