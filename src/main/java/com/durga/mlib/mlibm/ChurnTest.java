package com.durga.mlib.mlibm;

import org.apache.spark.ml.Model;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ChurnTest {

    private static StructType getTrainingSetSchema() {
        return new StructType(new StructField[] {
                new StructField("state", DataTypes.StringType,true, Metadata.empty()),
                new StructField("len", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("acode", DataTypes.StringType, true, Metadata.empty()),
                new StructField("intlplan", DataTypes.StringType, true, Metadata.empty()),
                new StructField("vplan", DataTypes.StringType, true, Metadata.empty()),
                new StructField("numvmail", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("tdmins", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("tdcalls", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("tdcharge", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("temins", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("tecalls", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("techarge", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("tnmins", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("tncalls", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("tncharge", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("timins", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("ticalls", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("ticharge", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("numcs", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("churn", DataTypes.StringType, true, Metadata.empty()),

        });
    }

    public static void main(String args[]){

        SparkSession spark = SparkSession
                .builder().master("local")
                .appName("JavaLogisticRegressionSummaryExample")
                .getOrCreate();


        String trainpath    =   "/media/durga/Other/ubuntu/data/churn-bigml-80.csv";
        String testpath     =   "/media/durga/Other/ubuntu/data/churn-bigml-20.csv";

        Dataset<Row> train = spark.read().schema(getTrainingSetSchema())
                .csv(trainpath);
        train.take(1);
        train.cache();
        System.out.println(train.count());

        Dataset<Row> test = spark.read().schema(getTrainingSetSchema())
                .csv(testpath);
        test.take(1);
        test.cache();
        System.out.println(test.count());
        test.printSchema();
        test.select("churn").show();

        train.createOrReplaceTempView("account");
        spark.catalog().cacheTable("account");

        Map<String,Double> fractions = new HashMap<>();
        fractions.put("False", 0.17d);
        fractions.put("True", 1d);
        Dataset<Row> strain = train.stat().sampleBy("churn",fractions ,36L );
        strain.groupBy("churn").count().show();

        Dataset<Row> ntrain = strain.drop("state").drop("acode").drop("vplan").drop("tdcharge").drop("techarge");
        ntrain.show();

        StringIndexer ipindexer = new StringIndexer()
                .setInputCol("intlplan")
                .setOutputCol("iplanIndex");

        StringIndexer labelindexer = new StringIndexer()
                .setInputCol("churn")
                .setOutputCol("label");

        List<String> featureCols =  Arrays.asList("len", "iplanIndex", "numvmail", "tdmins", "tdcalls", "temins", "tecalls", "tnmins", "tncalls", "timins", "ticalls", "numcs");

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(featureCols.toArray(new String[featureCols.size()]))
                .setOutputCol("features");

        DecisionTreeClassifier dTree = new DecisionTreeClassifier().setLabelCol("label")
                .setFeaturesCol("features");

        Pipeline pipeline = new Pipeline()
                .setStages(Arrays.asList(ipindexer, labelindexer, assembler, dTree).toArray(new PipelineStage[4]));

        ParamMap[] paramGrid = new ParamGridBuilder().
                addGrid(dTree.maxDepth(), new int[]{2, 3, 4, 5, 6, 7}).build();

        BinaryClassificationEvaluator evaluator = new BinaryClassificationEvaluator()
                .setLabelCol("label")
                .setRawPredictionCol("prediction");

        CrossValidator crossval = new CrossValidator().setEstimator(pipeline)
                .setEvaluator(evaluator)
                .setEstimatorParamMaps(paramGrid).setNumFolds(3);

        CrossValidatorModel cvModel =  crossval.fit(ntrain);

        Model<?> bestModel          = cvModel.bestModel();

        System.out.println(((DecisionTreeClassificationModel)(((PipelineModel)bestModel).stages()[3])).toDebugString() );

        Dataset<Row> predictions    =   cvModel.transform(test);
        double accuracy             =   evaluator.evaluate(predictions);
        evaluator.explainParams();

        System.out.println(predictions.select("prediction", "label").rdd().toString());

        predictions.show();


    }
}
