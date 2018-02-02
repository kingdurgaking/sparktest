package com.durga.contoller;

import com.durga.model.Count;
import org.apache.spark.sql.SparkSession;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
public class CustomLogisticRegression {

    @Autowired
    private SparkSession session;

    public void getRegression(){

        session.read().format("libsvm")
                .load("/media/durga/Other/ubuntu/Softwares/sparkexample/spark-master/data/mllib/sample_libsvm_data.txt");
    }


}
