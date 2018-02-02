package com.durga.mlib;


import java.util.Arrays;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Matrices;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.stat.Statistics;
import org.apache.spark.mllib.stat.test.ChiSqTestResult;
import org.apache.spark.sql.SparkSession;

public class HypothesisTest {

    public static void main(String args[]){

        SparkSession spark = SparkSession
                .builder().master("local")
                .appName("JavaLogisticRegressionSummaryExample")
                .getOrCreate();

        JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());

        Vector vec = Vectors.dense(0.1, 0.15, 0.2, 0.3, 0.25);

// compute the goodness of fit. If a second vector to test against is not supplied
// as a parameter, the test runs against a uniform distribution.
        ChiSqTestResult goodnessOfFitTestResult = Statistics.chiSqTest(vec);
// summary of the test including the p-value, degrees of freedom, test statistic,
// the method used, and the null hypothesis.
        System.out.println(goodnessOfFitTestResult + "\n");

// Create a contingency matrix ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
        Matrix mat = Matrices.dense(3, 2, new double[]{1.0, 3.0, 5.0, 2.0, 4.0, 6.0});

// conduct Pearson's independence test on the input contingency matrix
        ChiSqTestResult independenceTestResult = Statistics.chiSqTest(mat);
// summary of the test including the p-value, degrees of freedom...
        System.out.println(independenceTestResult + "\n");

// an RDD of labeled points
        JavaRDD<LabeledPoint> obs = jsc.parallelize(
                Arrays.asList(
                        new LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 3.0)),
                        new LabeledPoint(1.0, Vectors.dense(1.0, 2.0, 0.0)),
                        new LabeledPoint(-1.0, Vectors.dense(-1.0, 0.0, -0.5))
                )
        );

// The contingency table is constructed from the raw (feature, label) pairs and used to conduct
// the independence test. Returns an array containing the ChiSquaredTestResult for every feature
// against the label.
        ChiSqTestResult[] featureTestResults = Statistics.chiSqTest(obs.rdd());
        int i = 1;
        for (ChiSqTestResult result : featureTestResults) {
            System.out.println("Column " + i + ":");
            System.out.println(result + "\n");  // summary of the test
            i++;
        }

    }
}
