package com.example.handwritingcalculator;

import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.EditText;
import android.view.View;

import androidx.appcompat.app.AppCompatActivity;

import com.nex3z.fingerpaintview.FingerPaintView;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    // Our image size (MNIST uses 28x28)
    private static final int IMG_SIZE = 28;
    // there are 10 probabilities
    private static final int NUM_CLASSES = 10;
    // variables for xml widgets
    private FingerPaintView drawArea;
    private Button clearButton;
    private Button predictButton;
    private Button backButton;
    private EditText resultBox;
    private Button addButton;
    private Button minusButton;
    private Button multiplyButton;
    private Button divideButton;
    private Button equalButton;
    private final String TAG = "PHC";
    private Interpreter tflite;   // The ML model

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // connect layout items
        drawArea = findViewById(R.id.fpv);
        clearButton = findViewById(R.id.clearBtn);
        predictButton = findViewById(R.id.predictBtn);
        backButton = findViewById(R.id.backspaceBtn);
        resultBox = findViewById(R.id.textArea);
        addButton = findViewById(R.id.addBtn);
        minusButton = findViewById(R.id.minusBtn);
        multiplyButton = findViewById(R.id.multiplyBtn);
        divideButton = findViewById(R.id.divideBtn);
        equalButton = findViewById(R.id.equalBtn);

        try {
            tflite = new Interpreter(loadModel("digit.tflite"));
        } catch (Exception e) {
            tflite = null;
        }

        // clear drawing
        clearButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick (View view){
                drawArea.clear();
                resultBox.setText("");
            }
        });

        // remove last number
        backButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick (View view) {
                String oldText = resultBox.getText().toString();
                if (!oldText.isEmpty()) {
                    resultBox.setText(oldText.substring(0, oldText.length() - 1));
                }
            }
        });

        // appends a plus sign to the textArea
        addButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick (View view) {
                resultBox.append(" + ");
            }
        });

        // adds a minus sign
        minusButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick (View view) {
                resultBox.append(" - ");
            }
        });

        // add a multiplication symbol
        multiplyButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick (View view) {
                resultBox.append(" * ");
            }
        });

        // adds a division symbol
        divideButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick (View view) {
                resultBox.append(" / ");
            }
        });

        // equal symbol - solves the expression in textArea
        equalButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick (View view) {
                double answer = 0;
                String expression = resultBox.getText().toString();
                String[] newExpression = expression.split(" ");

                switch(newExpression[1]){
                    case "+":
                        answer = Double.parseDouble(newExpression[0]) + Double.parseDouble(newExpression[2]);
                        break;
                    case "-":
                        answer = Double.parseDouble(newExpression[0]) - Double.parseDouble(newExpression[2]);
                        break;
                    case "*":
                        answer = Double.parseDouble(newExpression[0]) * Double.parseDouble(newExpression[2]);
                        break;
                    case "/":
                        answer = Double.parseDouble(newExpression[0]) / Double.parseDouble(newExpression[2]);
                        break;
                }
                resultBox.append(" = " + answer);
            }
        });

        predictButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick (View view) {
                if (tflite == null) {
                    resultBox.setText("Model not loaded");
                    return;
                }

                // get drawing as a 28x28 picture
                Bitmap smallBitmap = drawArea.exportToBitmap(IMG_SIZE, IMG_SIZE);

                // turn the picture into numbers the model understands
                ByteBuffer inputBuffer = convertBitmapToByteBuffer(smallBitmap);

                // model output: 10 numbers (probabilities for digits 0–9)
                float[][] output = new float[1][NUM_CLASSES];

                // run the ML model
                tflite.run(inputBuffer, output);

                // find which digit has the highest score
                int digit = findBiggestValue(output[0]);

                try {
                    // show the digit
                    resultBox.append(String.valueOf(digit));
                } catch (Exception e) {
                    Log.v(TAG, e.getMessage());
                    resultBox.setText("Run error: " + e.getMessage());
                }
                // clear the drawing area
                drawArea.clear();
            }
        });
    }

    // load the model
    private MappedByteBuffer loadModel(String fileName) throws IOException {
        AssetFileDescriptor fileDescriptor = getAssets().openFd(fileName);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long length = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, length);
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bmp) {
        // the model needs 28*28 = 784 bytes
        // times 4 bytes since they are stored as floats
        ByteBuffer buffer = ByteBuffer.allocateDirect(IMG_SIZE * IMG_SIZE * 4);
        buffer.order(ByteOrder.nativeOrder());
        buffer.rewind();

        // loop through each pixel
        for (int y = 0; y < IMG_SIZE; y++) {
            for (int x = 0; x < IMG_SIZE; x++) {

                // read RGB
                int pixel = bmp.getPixel(x, y);
                float value;
                if(pixel == Color.WHITE){
                    value = 0;
                } else {
                    value = 1;
                }
                // put into the buffer as 1 byte
                buffer.putFloat(value);
            }
        }
        buffer.rewind();
        return buffer;
    }

    private int findBiggestValue(float[] list) {
        int index = 0;
        float max = list[0];

        for (int i = 1; i < list.length; i++) {
            if (list[i] > max) {
                max = list[i];
                index = i;
            }
        }
        return index;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (tflite != null) {
            tflite.close();
        }
    }
}