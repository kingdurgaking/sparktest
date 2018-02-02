package com.durga.contoller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;

/**
 * Created by Owner on 2017. 03. 29..
 */
@RequestMapping("api")
@Controller
public class ApiController {
    @Autowired
    CustomLogisticRegression regression;

    @RequestMapping("wordcount")
    public ResponseEntity<String> words() {
        regression.getRegression();
        return new ResponseEntity<>("Hello", HttpStatus.OK);
    }
}