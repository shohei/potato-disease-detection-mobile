package com.example.predictor

import android.content.Context
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.util.Arrays

class MainActivity : AppCompatActivity() {

    lateinit var module: Module

    override fun onCreate(savedInstanceState: Bundle?) {

        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        module = Module.load(assetFilePath(this, "model.pt"))

        var bitmap1 = BitmapFactory.decodeStream(getAssets().open("healthy.jpg"));
        val imView1: ImageView = findViewById<ImageView>(R.id.imageView1)
        imView1.setImageBitmap(bitmap1);
        imView1.setOnClickListener(View.OnClickListener() {
            predict(bitmap1);
        })
        var bitmap2 = BitmapFactory.decodeStream(getAssets().open("early.jpg"));
        val imView2: ImageView = findViewById<ImageView>(R.id.imageView2)
        imView2.setImageBitmap(bitmap2);
        imView2.setOnClickListener(View.OnClickListener() {
            predict(bitmap2);
        })
        var bitmap3 = BitmapFactory.decodeStream(getAssets().open("late.jpg"));
        val imView3: ImageView = findViewById<ImageView>(R.id.imageView3)
        imView3.setImageBitmap(bitmap3);
        imView3.setOnClickListener(View.OnClickListener() {
            predict(bitmap3);
        })

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

    private fun predict(bitmap: Bitmap) {
        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB)
        val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()
        val scores = outputTensor.dataAsFloatArray
        val arg = argmax(scores)
        val labels = arrayOf("Early blight", "Healthy", "Late blight")
        val tv: TextView = findViewById<TextView>(R.id.textView)
        tv.setText(Arrays.toString(scores))
        Log.v("hoge",Arrays.toString(scores))
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
