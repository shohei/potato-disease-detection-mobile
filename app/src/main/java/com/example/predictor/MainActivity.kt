package com.example.predictor

import android.content.Context
import android.graphics.*
import android.os.Bundle
import android.widget.Button
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream

class MainActivity : AppCompatActivity() {

    lateinit var module: Module

    override fun onCreate(savedInstanceState: Bundle?) {

        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val predButton:Button = findViewById<Button>(R.id.predict_button)
        predButton.setOnClickListener{
            predict()
        }

        module = Module.load(assetFilePath(this, "model.pt"))
    }
    fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }
        context.assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
            return file.absolutePath
        }
    }

    private fun predict() {
        var bitmap = BitmapFactory.decodeStream(getAssets().open("leaf.jpg"));
        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB)
        val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()
        val scores = outputTensor.dataAsFloatArray
        val arg = argmax(scores)
        val labels = arrayOf("Early blight", "Healthy", "Late blight")
        Toast.makeText(applicationContext, labels[arg], Toast.LENGTH_SHORT).show();
    }

    fun argmax(array: FloatArray): Int {
        var max = array[0]
        var re = 0
        for (i in 1 until array.size) {
            if (array[i] > max) {
                max = array[i]
                re = i
            }
        }
        return re
    }


}
