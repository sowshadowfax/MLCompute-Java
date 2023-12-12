import ANNCompute.*;

public class test {
    public static void main(String[] args) {
        double[] x = {1, 2, 3, 4, 5};
        double[] y = {2, 4, 6, 8, 10};
        double b = 0.0;
        double lr = 0.01;

        double epochs = 5;

        /*
        // Demo 1: Simple Linear Regressor
        SimpleLinearRegressor slr = new SimpleLinearRegressor();
        slr.initialize(b, x, y, lr);

        // Training loop
        for (int i = 0; i < epochs; i++) {
            slr.forwardPass();
            slr.simpleGradient();
            System.out.println("Loss: " + slr.squaredError());
        }

        double[] x_t = {6, 7, 8, 9, 10};
        double[] y_t = slr.predict(x_t);

        for (int i = 0; i < x_t.length; i++)
            System.out.println("Predicted: " + y_t[i]); */
        
        /*
        // Demo 2: Multi Linear Regressor
        double[] x2 = {1,2,3,4,5};
        double[] y2 = {1,4,9,16,25};

        MultiLinearRegressor mlr = new MultiLinearRegressor();
        mlr.initialize(b, x2, y2, lr);

        // Training loop
        for (int i = 0; i < epochs; i++) {
            mlr.forwardPass();
            mlr.simpleGradient();
            System.out.println("Loss: " + mlr.squaredError());
        }

        double[] x_t2 = {6, 7, 8, 9, 10};
        double[] y_t2 = mlr.predict(x_t2);

        for (int i = 0; i < x_t2.length; i++)
            System.out.println("Predicted: " + y_t2[i]); */
        
        /*
        // Demo 3: Linear Regressor
        double[][] x3 = {{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}};
        double[] y3 = {3, 7, 11, 15, 19};

        LinearRegressor lir = new LinearRegressor();
        lir.initialize(b, x3, y3, lr);

        // Training loop
        for (int i = 0; i < epochs; i++) {
            lir.forwardPass();
            lir.simpleGradient();
            System.out.println("Loss: " + lir.squaredError());
        }

        double[][] x_t3 = {{11, 12}, {13, 14}, {15, 16}, {17, 18}, {19, 20}};
        double[] y_t3 = lir.predict(x_t3);

        for (int i = 0; i < x_t3.length; i++)
            System.out.println("Predicted: " + y_t3[i]); */
        
        // Demo 4: Simple Linear Classifier
        double[] x4 = {1, 2, 3, 4, 5};
        double[] y4 = {0, 0, 1, 1, 1};

        SimpleLinearClassifier slc = new SimpleLinearClassifier();
        slc.initialize(b, x4, y4, lr);

        // Training loop
        for (int i = 0; i < epochs; i++) {
            slc.forwardPass();
            slc.simpleGradient();
            System.out.println("Loss: " + slc.squaredError());
        }

        double[] x_t4 = {0, 1, 2, 3, 4, 5};
        int[] y_t4 = slc.predict(x_t4);

        for (int i = 0; i < x_t4.length; i++)
            System.out.println("Predicted: " + y_t4[i]);
        
    }
}
