package com.example.gesturedetection;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import android.Manifest;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.FrameLayout;
import android.widget.ImageView;
import static com.example.gesturedetection.R.id.action_open;
import static com.example.gesturedetection.R.id.action_detect;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class MainActivity extends AppCompatActivity {

    static final int REQUEST_OPEN_IMAGE = 1;
    String[] PERMISSIONS = {
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.READ_EXTERNAL_STORAGE
    };

    private ImageView imageView;
    private FrameLayout loadingScreen;
    private Interpreter interpreter;
    private Bitmap signImage;
    private Bitmap bitmap;
    private String currentPhotoPath = "";
    private List<String> labels;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        ActivityCompat.requestPermissions(this,
                PERMISSIONS,
                1);

        imageView = (ImageView) findViewById(R.id.image);
        loadingScreen = (FrameLayout) findViewById(R.id.vg_loading);

        try {
            long timeStart = System.currentTimeMillis();
            interpreter = new Interpreter(loadModelFile());
            long timeEnd = System.currentTimeMillis();
            System.out.println("Load: " + (timeEnd - timeStart));
        } catch (Exception e) {
            e.printStackTrace();
        }



    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        switch (item.getItemId()) {
            case action_open:
                Intent getPicture = new Intent(Intent.ACTION_GET_CONTENT);
                getPicture.setType("image/*");
                Intent pickPicture = new Intent(Intent.ACTION_PICK,
                        MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                Intent chooser = Intent.createChooser(getPicture, "Selectati Imaginea");
                chooser.putExtra(Intent.EXTRA_INITIAL_INTENTS, new Intent[] {
                        pickPicture
                });
                startActivityForResult(chooser, REQUEST_OPEN_IMAGE);
                return true;
            case action_detect:
                predictGesture();
                return true;
        }
        return false;
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        switch (requestCode) {
            case REQUEST_OPEN_IMAGE:
                if (resultCode == RESULT_OK) {
                    Uri imgUri = data.getData();
                    String[] filePathColumn = {MediaStore.Images.Media.DATA};
                    Cursor cursor = getContentResolver().query(imgUri, filePathColumn,
                            null, null, null);
                    cursor.moveToFirst();

                    int colIndex = cursor.getColumnIndex(filePathColumn[0]);
                    currentPhotoPath = cursor.getString(colIndex);
                    cursor.close();
                    setImageFromFilePath();
                }
                break;
        }
    }

    private void setImageFromFilePath() {
        BitmapFactory.Options bitmapOptions = new BitmapFactory.Options();
        bitmapOptions.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(currentPhotoPath, bitmapOptions);

        bitmapOptions.inJustDecodeBounds = false;
        bitmapOptions.inPurgeable = true;

        bitmap = BitmapFactory.decodeFile(currentPhotoPath, bitmapOptions);
        imageView.setImageBitmap(bitmap);
    }

    public void showMessage(String title,String Message){
        AlertDialog.Builder builder=new AlertDialog.Builder(this);
        builder.setCancelable(true);
        builder.setTitle(title);
        builder.setMessage(Message);
        builder.show();
    }

    private void predictGesture() {

        loadingScreen.setVisibility(View.VISIBLE);
        int imageTensorIndex = 0;
        int[] imageShape = interpreter.getInputTensor(imageTensorIndex).shape();
        int imageSizeY = imageShape[1];
        int imageSizeX = imageShape[2];
        DataType imageDataType = interpreter.getInputTensor(imageTensorIndex).dataType();

        int probabilityTensorIndex = 0;
        int[] probabilityShape =
                interpreter.getOutputTensor(probabilityTensorIndex).shape();
        DataType probabilityDataType = interpreter.getOutputTensor(probabilityTensorIndex).dataType();

        TensorImage inputImageBuffer = new TensorImage(imageDataType);
        TensorBuffer outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);
        TensorProcessor probabilityProcessor = new TensorProcessor.Builder().add(new NormalizeOp(0.0f, 255.0f)).build();
        signImage = BitmapFactory.decodeFile(currentPhotoPath);
        Bitmap scaledImage = Bitmap.createScaledBitmap(signImage, imageSizeX, imageSizeY, true);

        if(imageShape[imageShape.length - 1] == 3) {
            inputImageBuffer.load(signImage);

            ImageProcessor imageProcessor =
                    new ImageProcessor.Builder()
                            .add(new ResizeWithCropOrPadOp(imageSizeX, imageSizeY))
                            .add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                            .add(new NormalizeOp(0.0f, 1.0f))
                            .build();
            inputImageBuffer = imageProcessor.process(inputImageBuffer);
            long timeStart = System.currentTimeMillis();
            interpreter.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());
            long timeEnd = System.currentTimeMillis();
            System.out.println("Predict: " + (timeEnd - timeStart));
        } else {
            byte[] grayImage = convertToGray(scaledImage);
            ByteBuffer imgData = ByteBuffer.allocateDirect(4 * imageSizeX * imageSizeY);
            imgData.order(ByteOrder.nativeOrder());
            imgData.rewind();
            imgData.put(grayImage);
            long timeStart = System.currentTimeMillis();
            interpreter.run(imgData, outputProbabilityBuffer.getBuffer().rewind());
            long timeEnd = System.currentTimeMillis();
            System.out.println("Predict: " + (timeEnd - timeStart));
        }

        try {
            labels = FileUtil.loadLabels(MainActivity.this, "labels.txt");
        } catch (Exception e) {
            e.printStackTrace();
        }
        Map<String, Float> labeledProbability =
                new TensorLabel(labels, probabilityProcessor.process(outputProbabilityBuffer))
                        .getMapWithFloatValue();
        float maxValueInMap = (Collections.max(labeledProbability.values()));

        for (Map.Entry<String, Float> value : labeledProbability.entrySet()) {
            if(value.getValue() == maxValueInMap) {
                showMessage("Predictia", value.getKey());
            }
        }
        loadingScreen.setVisibility(View.INVISIBLE);
    }

    private byte[] convertToGray(Bitmap bmp) {
        int width = bmp.getWidth();
        int height = bmp.getHeight();
        byte[] grayscale = new byte[width * height];
        int val, R, G, B;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                val = bmp.getPixel(j, i);
                R = (val >> 16) & 0xff;
                G = (val >> 8) & 0xff;
                B = val & 0xff;
                grayscale[i * width + j] = (byte) (0.21 * R + 0.71 * G + 0.07 * B);
            }
        }

        return grayscale;
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
}