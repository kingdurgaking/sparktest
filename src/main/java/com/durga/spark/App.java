package com.durga.spark;

import com.durga.contoller.CustomLogisticRegression;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.BinaryLogisticRegressionSummary;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionTrainingSummary;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/*
@SpringBootApplication
@ComponentScan({"com.durga.*"})
*/
public class App {


    private static StructType getTrainingSetSchema() {
        return new StructType(new StructField[] {
                new StructField("Product", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("Price", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("Name", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("City", DataTypes.IntegerType, true, Metadata.empty()),
        });
    }

    public static void main(String[] args) {
      //  SpringApplication.run(App.class, args);


        SparkSession spark = SparkSession
                .builder().master("local")
                .appName("JavaLogisticRegressionSummaryExample")
                .getOrCreate();

        Dataset<Row> trainingSet = spark.read().option("header", "true").schema(getTrainingSetSchema())
                .csv("/media/durga/Other/ubuntu/data/data.csv");


        LinearRegression linearRegression = new LinearRegression();

        Pipeline pipeline = new Pipeline().setStages(
                new PipelineStage[]{
                        linearRegression});

        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(linearRegression.regParam(), new double[] {0.3, 0.8})
                .addGrid(linearRegression.fitIntercept())
                .addGrid(linearRegression.elasticNetParam(), new double[] {0.3, 0.8})
                .addGrid(linearRegression.maxIter(), new int[] {50, 100})
                .build();

        // CrossValidator will try all combinations of values and determine best model using the evaluator.
        CrossValidator crossValidator = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(new RegressionEvaluator())
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(10);

        // Run cross-validation, and choose the best set of parameters.
        CrossValidatorModel model = crossValidator.fit(trainingSet);


        Dataset<Row> features = spark.createDataFrame(
                Collections.singletonList(RowFactory.create(1100, "Visa", "New York")),
                new StructType(new StructField[]{
                        DataTypes.createStructField("Price", DataTypes.IntegerType, false),
                        DataTypes.createStructField("Payment_Type", DataTypes.StringType, false),
                        DataTypes.createStructField("City", DataTypes.StringType, false),
                }));

        getRegressionPipelineStatistics(model);

        System.out.println((String) model.bestModel().transform(features).first().getAs("prediction"));
    }

    private static Map<String, Object> getRegressionPipelineStatistics(CrossValidatorModel model) {
        Map<String, Object> modelStatistics = new HashMap<>();

        Arrays.sort(model.avgMetrics());
        modelStatistics.put("Best avg metrics", model.avgMetrics()[model.avgMetrics().length - 1]);

        LinearRegressionModel linearRegressionModel = (LinearRegressionModel) ((PipelineModel) model.bestModel()).stages()[1];
        modelStatistics.put("Best intercept", linearRegressionModel.intercept());
        modelStatistics.put("Best max iterations", linearRegressionModel.getMaxIter());
        modelStatistics.put("Best reg parameter", linearRegressionModel.getRegParam());
        modelStatistics.put("Best elastic net parameter", linearRegressionModel.getElasticNetParam());

        Map<String, Object> coefficientsExplained = new HashMap<>();
        double[] coefficients = linearRegressionModel.coefficients().toArray();

        coefficientsExplained.put("Price", coefficients[0]);
        coefficientsExplained.put("City", coefficients[1]);

     /*   for (int i = 2; i < coefficients.length; i++) {
            ProgrammingLanguage programmingLanguage = DouConverter.transformProgrammingLanguage(i-2);
            coefficientsExplained.put("Language " + programmingLanguage.getName(), coefficients[i]);
        }*/

        modelStatistics.put("Coefficients", coefficientsExplained);


        return modelStatistics;
    }
}