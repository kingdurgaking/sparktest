package com.durga.spark;

import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.catalyst.encoders.RowEncoder;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.io.IOException;
import java.util.UUID;

public class CustomFeatureExtractor extends Transformer implements MLWritable {

    private String uid;

    public CustomFeatureExtractor(){
        this.uid = "DouRegressionFeatureExtractor" + "_" + UUID.randomUUID().toString();
    }

    @Override
    public Dataset<Row> transform(Dataset<?> dataset) {
        Dataset<Row> ds = dataset.select("Product", "price", "Name", "City");



        //dataset = dataset.map(new CustomRegression(),  RowEncoder.apply(transformSchema(dataset.schema())));

        ds.count();
        ds.cache();

        return ds;
    }

    @Override
    public StructType transformSchema(StructType schema) {
        return new StructType(new StructField[]{
                DataTypes.createStructField("label", DataTypes.DoubleType, false),
                DataTypes.createStructField("features", new VectorUDT(), false)});
    }

    @Override
    public Transformer copy(ParamMap extra) {
        return super.defaultCopy(extra);
    }

    @Override
    public String uid() {
        return this.uid;
    }

    @Override
    public MLWriter write() {
        return new DefaultParamsWriter(this);
    }

    @Override
    public void save(String path) throws IOException {
        write().save(path);
    }

    public static MLReader<CustomFeatureExtractor> read() {
        return new DefaultParamsReader<>();
    }
}
