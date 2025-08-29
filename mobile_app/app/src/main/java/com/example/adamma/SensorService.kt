package com.example.adamma

import android.app.Notification
import android.app.Service
import android.app.NotificationChannel
import android.content.pm.ServiceInfo
import android.app.NotificationManager
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import kotlin.math.sqrt


class SensorService : Service(), SensorEventListener {

    private lateinit var sensorManager: SensorManager
    private lateinit var dataStore: DataStore

    private var lastAccel: FloatArray? = null
    private var lastGyro: FloatArray? = null

    // (timestamp, row)
    private val window = mutableListOf<Pair<Long, FloatArray>>()
    private val WINDOW_SIZE = 150 // ~3s @ 50Hz, but we’ll measure elapsed time

    private fun ensureChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val nm = getSystemService(NotificationManager::class.java)

            val ch = NotificationChannel(
                "activity_tracking_channel_v2",  // 👈 new ID
                "Activity Tracking",
                NotificationManager.IMPORTANCE_DEFAULT
            ).apply {
                setShowBadge(false)
                lockscreenVisibility = Notification.VISIBILITY_PUBLIC
            }
            nm.createNotificationChannel(ch)
        }
    }





    override fun onCreate() {
        super.onCreate()

        ensureChannel() // ✅ must be before building the notification

        val notification = NotificationCompat.Builder(this, "activity_tracking_channel_v2")
            .setContentTitle("ADAMMA Activity Tracking")
            .setContentText("Tracking your activity in background")
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setOngoing(true)
            .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE) // 👈
            .build()


        if (Build.VERSION.SDK_INT >= 34) {
            startForeground(
                1,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_HEALTH or
                        ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
            )
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(1, notification, 0)
        } else {
            startForeground(1, notification)
        }

        dataStore = DataStore.getInstance(this)

        Scaler.load(this)
        ModelInference.init(this)

        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        val accel = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        val gyro  = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        sensorManager.registerListener(this, accel, 20_000)
        sensorManager.registerListener(this, gyro, 20_000)
    }


    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> lastAccel = event.values.clone()
            Sensor.TYPE_GYROSCOPE -> lastGyro = event.values.clone()
        }

        if (lastAccel != null && lastGyro != null) {
            val ax = lastAccel!![0] / 9.81f
            val ay = lastAccel!![1] / 9.81f
            val az = lastAccel!![2] / 9.81f
            val gx = lastGyro!![0]
            val gy = lastGyro!![1]
            val gz = lastGyro!![2]

            val amag = sqrt(ax*ax + ay*ay + az*az)
            val gmag = sqrt(gx*gx + gy*gy + gz*gz)

            val row = floatArrayOf(ax, ay, az, gx, gy, gz, amag, gmag)
            window.add(event.timestamp to row)

            if (window.size >= WINDOW_SIZE) {
                try {
                    // Flatten raw window [150,8] → [1200]
                    val rawFlat = FloatArray(window.size * 8)
                    var idx = 0
                    for ((_, r) in window) {
                        for (v in r) rawFlat[idx++] = v
                    }

                    // Extract engineered features [36]
                    val feat = FeatureExtractor.extract(window.map { it.second })

                    // Scale inputs
                    val rawScaled = Scaler.scaleRaw(rawFlat)
                    val featScaled = Scaler.scaleFeat(feat)

                    // Run prediction
                    val metClass = ModelInference.predict(rawScaled, WINDOW_SIZE, featScaled)

                    // Measure elapsed time from sensor timestamps
                    val elapsedSec = ((window.last().first - window.first().first) / 1_000_000_000L)
                        .toInt().coerceAtLeast(1)

                    dataStore.updateClass(metClass, elapsedSec)

                    Log.d("SensorService", "Predicted: $metClass ($elapsedSec s)")
                } catch (e: Exception) {
                    Log.e("SensorService", "Prediction failed", e)
                } finally {
                    window.clear()
                }
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        super.onDestroy()
        sensorManager.unregisterListener(this)

        // Stop foreground and remove notification (modern API)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            stopForeground(STOP_FOREGROUND_REMOVE)
        } else {
            @Suppress("DEPRECATION")
            stopForeground(true)
        }

        stopSelf()
    }

}
