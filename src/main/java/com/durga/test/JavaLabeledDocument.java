package com.durga.test;

import java.io.Serializable;

public class JavaLabeledDocument extends JavaDocument implements Serializable {

    private double label;

    public JavaLabeledDocument(long id, String text, double label) {
        super(id, text);
        this.label = label;
    }

    public double getLabel() {
        return this.label;
    }
}
