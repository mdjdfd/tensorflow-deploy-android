package com.open.androidmlhandsdirty

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import androidx.databinding.DataBindingUtil
import com.open.androidmlhandsdirty.databinding.ActivityMainBinding
import com.open.androidmlhandsdirty.ml.Irismodel
import kotlinx.android.synthetic.main.activity_main.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer

class MainActivity : AppCompatActivity() {
    lateinit var activityMainBinding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        activityMainBinding = DataBindingUtil.setContentView(this, R.layout.activity_main)
        activityMainBinding.activity = this
    }


    fun onPredictClick(view: View) {
        var byteBuffer: ByteBuffer = ByteBuffer.allocate(4 * 4)
        byteBuffer.putFloat(textinputedittext_sepal_length.text.toString().toFloat())
        byteBuffer.putFloat(textinputedittext_sepal_width.text.toString().toFloat())
        byteBuffer.putFloat(textinputedittext_petal_length.text.toString().toFloat())
        byteBuffer.putFloat(textinputedittext_petal_width.text.toString().toFloat())
        predictFromModel(byteBuffer)
    }


    private fun predictFromModel(byteBuffer: ByteBuffer) {
        val model = Irismodel.newInstance(this)

        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 4), DataType.FLOAT32)
        inputFeature0.loadBuffer(byteBuffer)

        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

        textview_prediction_result.text =
            "Iris-setosa : " + outputFeature0[0].toString() + "\n" + "Iris-versicolor : " + outputFeature0[1].toString() + "\n" + "Iris-verginica : " + outputFeature0[2].toString()

        model.close()
    }
}