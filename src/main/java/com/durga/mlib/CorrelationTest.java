package com.durga.mlib;

import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.stat.Statistics;
import org.apache.spark.sql.SparkSession;

import java.util.Arrays;

public class CorrelationTest {
    public static void main(String args[]){
        SparkSession spark = SparkSession
                .builder().master("local")
                .appName("JavaLogisticRegressionSummaryExample")
                .getOrCreate();

        JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());
        JavaDoubleRDD seriesX = jsc.parallelizeDoubles(
                Arrays.asList(1.0, 2.0, 3.0, 3.0, 5.0));  // a series

// must have the same number of partitions and cardinality as seriesX
        JavaDoubleRDD seriesY = jsc.parallelizeDoubles(
                Arrays.asList(11.0, 22.0, 33.0, 33.0, 555.0));

// compute the correlation using Pearson's method. Enter "spearman" for Spearman's method.
// If a method is not specified, Pearson's method will be used by default.
        Double correlation = Statistics.corr(seriesX.srdd(), seriesY.srdd(), "pearson");
        System.out.println("Correlation is: " + correlation);



        JavaRDD<Vector> data = jsc.parallelize(
                Arrays.asList(
                        Vectors.dense(1.0, 10.0, 100.0),
                        Vectors.dense(2.0, 20.0, 200.0),
                        Vectors.dense(5.0, 33.0, 366.0)
                )
        );

// calculate the correlation matrix using Pearson's method.
// Use "spearman" for Spearman's method.
// If a method is not specified, Pearson's method will be used by default.
        Matrix correlMatrix = Statistics.corr(data.rdd(), "pearson");
        System.out.println(correlMatrix.toString());

    }
}
