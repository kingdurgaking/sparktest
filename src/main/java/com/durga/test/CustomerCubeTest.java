package com.durga.test;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.RelationalGroupedDataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.storage.StorageLevel;

public class CustomerCubeTest {

    private static StructType getTrainingSetSchema() {
        return new StructType(new StructField[] {
                new StructField("Cifid", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("TranDate", DataTypes.DateType, true, Metadata.empty()),
                new StructField("Vendor", DataTypes.StringType, true, Metadata.empty()),
                new StructField("Amount", DataTypes.DoubleType, true, Metadata.empty()),
        });
    }

    public static void main(String args[]){
        SparkSession spark = SparkSession
                .builder().master("local")
                .appName("JavaLogisticRegressionSummaryExample")
                .getOrCreate();

        Dataset<Row> dfs = spark.read().option("header", "true").schema(getTrainingSetSchema()).format("csv")
                .option("dateFormat","dd-MMM-yyyy" )
                .load("/media/durga/Other/ubuntu/data/datafile.csv");

       // dfs.groupBy("Cifid","TranDate","Vendor").sum("Amount").as("Amount Spent").show();

        RelationalGroupedDataset dm =   dfs.cube("Cifid","TranDate","Vendor");
        Dataset<Row> ds = dm.agg(org.apache.spark.sql.functions.sum("Amount"),
                org.apache.spark.sql.functions.count("Amount"),
                org.apache.spark.sql.functions.max("Amount"),
                org.apache.spark.sql.functions.min("Amount"))
               .withColumnRenamed("sum(Amount)", "Total Amount")
               .withColumnRenamed("count(Amount)", "Count")
               .withColumnRenamed("max(Amount)", "Max Amount")
               .withColumnRenamed("min(Amount)", "Min Amount");


        StorageLevel sl= new StorageLevel();
        ds.persist(sl);//.start("/media/durga/Other/ubuntu/data/cubedb/livdb.cube");

        ds.show();


        ds.filter(" cifid=100 and Vendor = 'Spinnys' and TranDate is null ").show();
        //dm.agg(dfs.col("Product")).filter("Product==2").show();

        //dm.agg(dfs.col("City")).printSchema();
       // dm.pivot("Product").count().show();
    }

}
