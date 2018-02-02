package com.durga.mlib;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary;
import org.apache.spark.mllib.stat.Statistics;
import org.apache.spark.sql.SparkSession;

import java.util.Arrays;

public class SummaryStatsTest {

    public static void main(String args[]){


        SparkSession spark = SparkSession
                .builder().master("local")
                .appName("JavaLogisticRegressionSummaryExample")
                .getOrCreate();

        JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());
       JavaRDD<Vector> mat = jsc.parallelize(Arrays.asList(
                Vectors.dense(1.0, 10.0, 100.0),
                Vectors.dense(2.0, 20.0, 200.0),
                Vectors.dense(3.0, 30.0, 300.0)
        ));

       /* JavaRDD<Vector> mat = jsc.parallelize(Arrays.asList(
                Vectors.dense(1.0, 10.0, 100.0,
                2.0, 20.0, 200.0,
                3.0, 30.0, 300.0)
        ));*/

        MultivariateStatisticalSummary summary = Statistics.colStats(mat.rdd());
        System.out.println(summary.mean());  // a dense vector containing the mean value for each column
        System.out.println(summary.variance());  // column-wise variance
        System.out.println(summary.numNonzeros());  // number of nonzeros in each column
        // $example off$

        jsc.stop();

    }
}
