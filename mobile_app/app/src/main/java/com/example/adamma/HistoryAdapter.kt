package com.example.adamma

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView

class HistoryAdapter : RecyclerView.Adapter<HistoryAdapter.HistoryViewHolder>() {

    private var data: Map<String, Map<String, Int>> = emptyMap()

    fun setData(newData: Map<String, Map<String, Int>>) {
        data = newData.toSortedMap(compareByDescending { it })
        notifyDataSetChanged()
    }

    class HistoryViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val dateText: TextView = view.findViewById(R.id.dateText)
        val summaryText: TextView = view.findViewById(R.id.summaryText)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): HistoryViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_history, parent, false)
        return HistoryViewHolder(view)
    }

    override fun onBindViewHolder(holder: HistoryViewHolder, position: Int) {
        val entry = data.entries.toList()[position]
        val date = entry.key
        val breakdown = entry.value

        holder.dateText.text = date
        holder.summaryText.text = breakdown.entries.joinToString(" | ") { (cls, secs) ->
            val minutes = secs / 60
            val remSec = secs % 60
            if (minutes > 0) "$cls: ${minutes}m ${remSec}s" else "$cls: ${remSec}s"
        }
    }

    override fun getItemCount(): Int = data.size
}
