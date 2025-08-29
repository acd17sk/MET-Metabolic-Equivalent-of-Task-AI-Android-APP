package com.example.adamma

import android.Manifest
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.widget.Button
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.github.mikephil.charting.charts.PieChart
import com.google.android.material.textview.MaterialTextView

class MainActivity : AppCompatActivity() {

    private lateinit var currentClassText: MaterialTextView
    private lateinit var pieChart: PieChart
    private lateinit var historyButton: Button
    private lateinit var startButton: Button
    private lateinit var stopButton: Button

    private lateinit var refreshButton: Button
    private lateinit var dataStore: DataStore

    // ✅ Android runtime permissions
    private val requestPermissionsLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val granted = permissions.all { it.value }
        if (granted) {
            startSensorService()
        } else {
            currentClassText.text = "Permission denied. Cannot access sensors."
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // 🔔 Create notification channel for SensorService
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                "activity_channel",
                "Activity Tracking",
                NotificationManager.IMPORTANCE_LOW
            )
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(channel)
        }

        // Init ONNX model
        ModelInference.init(this)

        // UI
        currentClassText = findViewById(R.id.currentClassText)
        pieChart = findViewById(R.id.pieChart)
        historyButton = findViewById(R.id.historyButton)
        startButton = findViewById(R.id.startButton)
        stopButton = findViewById(R.id.stopButton)
        refreshButton = findViewById(R.id.refreshButton)
        dataStore = DataStore.getInstance(this)

        // Request runtime permissions
        checkAndRequestPermissions()

        // Observe classification updates
        dataStore.metClassLive.observe(this) { metClass ->
            currentClassText.text = "Current Activity: $metClass"
            pieChart.data = dataStore.getPieData()
            pieChart.invalidate()
        }

        // Navigate to history
        historyButton.setOnClickListener {
            startActivity(Intent(this, HistoryActivity::class.java))
        }

        // Start tracking manually
        startButton.setOnClickListener {
            startSensorService()
        }

        // Stop tracking manually
        stopButton.setOnClickListener {
            stopService(Intent(this, SensorService::class.java))
            currentClassText.text = "Tracking stopped"

            // Clear notification manually
            val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
            nm.cancel(1) // ID must match startForeground()
        }

        refreshButton.setOnClickListener {
            dataStore.resetData()
            pieChart.data = dataStore.getPieData()
            pieChart.invalidate()
        }




    }

    // ✅ Request Android runtime permissions
    private fun checkAndRequestPermissions() {
        val neededPermissions = mutableListOf<String>()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACTIVITY_RECOGNITION)
                != PackageManager.PERMISSION_GRANTED
            ) {
                neededPermissions.add(Manifest.permission.ACTIVITY_RECOGNITION)
            }
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED) {
                neededPermissions.add(Manifest.permission.POST_NOTIFICATIONS)
            }
        }


        if (ContextCompat.checkSelfPermission(this, Manifest.permission.BODY_SENSORS)
            != PackageManager.PERMISSION_GRANTED
        ) {
            neededPermissions.add(Manifest.permission.BODY_SENSORS)
        }

        if (neededPermissions.isNotEmpty()) {
            requestPermissionsLauncher.launch(neededPermissions.toTypedArray())
        } else {
            startSensorService()
        }
    }

    private fun startSensorService() {
        startService(Intent(this, SensorService::class.java))
    }
}
