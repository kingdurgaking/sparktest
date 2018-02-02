package com.durga.test;

import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class CubeTest {

    private static StructType getTrainingSetSchema() {
        return new StructType(new StructField[] {
                new StructField("Product", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("Price", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("Name", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("City", DataTypes.IntegerType, true, Metadata.empty()),
        });
    }

    public static void main(String args[]){
        SparkSession spark = SparkSession
                .builder().master("local")
                .appName("JavaLogisticRegressionSummaryExample")
                .getOrCreate();

        Dataset<Row> dfs = spark.read().option("header", "true").schema(getTrainingSetSchema())
                .csv("/media/durga/Other/ubuntu/data/Mins.csv");

/*
        Column prodCol      =   dfs.col("Product");
        Column price        =   dfs.col("Price");
        Column name         =   dfs.col("Name");
        Column city         =   dfs.col("City");*/

        /*RelationalGroupedDataset dm =   df.cube("Product" , "Price" , "Name" , "City");
        dm.count().show();*/
/*
        Dataset<Row> newdf  = spark.read().option("header", "true").schema(getTrainingSetSchema())
                .csv("/media/durga/Other/ubuntu/data/bs.csv");

        dfs                 =   dfs.union(newdf);*/


     //   Dataset<Row> dr        =    dfs.filter(dfs.col("Product").isNotNull());//.filter(dfs.col("Product").equalTo("3"));

        RelationalGroupedDataset dm =   dfs.cube("Product","Price");

        dm.count().show();
        //dm.agg(dfs.col("Product")).filter("Product==2").show();

        //dm.agg(dfs.col("City")).printSchema();
       // dm.pivot("Product").count().show();
    }

}
