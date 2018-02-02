package com.durga.mlib;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import org.spark_project.guava.collect.ImmutableMap;
import scala.Tuple2;

import java.util.Arrays;
import java.util.List;

public class StratifiedSampTest {
    public static void main(String args[]){
        SparkSession spark = SparkSession
                .builder().master("local")
                .appName("JavaLogisticRegressionSummaryExample")
                .getOrCreate();

        JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());


        List<Tuple2<Double, Character>> list = Arrays.asList(
                new Tuple2<>(1d, 'a'),
                new Tuple2<>(1d, 'b'),
                new Tuple2<>(2d, 'c'),
                new Tuple2<>(2d, 'd'),
                new Tuple2<>(2d, 'e'),
                new Tuple2<>(3d, 'f')
        );

        JavaPairRDD<Double, Character> data = jsc.parallelizePairs(list);

// specify the exact fraction desired from each key Map<K, Double>
        ImmutableMap<Double, Double> fractions = ImmutableMap.of(1d, 0.1d, 2d, 0.6d, 3d, 0.3d);

// Get an approximate sample from each stratum
        JavaPairRDD<Double, Character> approxSample = data.sampleByKey(false, fractions);
// Get an exact sample from each stratum
        JavaPairRDD<Double, Character> exactSample = data.sampleByKeyExact(false, fractions);

        System.out.println(approxSample.collectAsMap());

        System.out.println(exactSample.collectAsMap());
    }
}
