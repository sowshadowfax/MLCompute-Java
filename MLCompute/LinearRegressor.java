package MLCompute;
import java.util.Random;
import java.math.*;

public class LinearRegressor {
    
        // Declaring variables for the equation y = w1x1 + w2x1 + ... + wnx1 + b
        double[] w = {};
        double[][] x;
        double b = 0.0;
        double[] y = {};
        double[] y_pred = {};
    
        // Declaring variables for gradient descent / back-propagation
        double[] grad = {};
        double lr = 0.01; // Learning rate
        double loss = 0.0;
    
        // Initializes the weights randomly and assigns bias
        public void initialize(double b, double[][] x, double[] y, double lr) {
            Random rand = new Random();

            try {
            this.w = new double[x[0].length];
            } catch (Exception e) {
                System.out.println(e);
                System.out.println("Caught Error while initializing weights in LinearRegressor");
                return;
            }

            try {
                for (int i = 0; i < this.w.length; i++)
                    this.w[i] = rand.nextDouble();
            } catch(Exception e){
                System.out.println(e);
                System.out.println("Caught Error while allocating weights in LinearRegressor");
                return;
            }

            try {
            this.b = b;
            } catch (Exception e) {
                System.out.println(e);
                System.out.println("Caught Error while initializing bias in LinearRegressor");
                return;
            }

            try {
            this.x = new double[x.length][x[0].length];
            } catch (Exception e) {
                System.out.println(e);
                System.out.println("Caught Error while initializing x in LinearRegressor");
                return;
            }

            try {
            for(int i = 0; i < x.length; i++)
                this.x[i] = x[i];
            } catch (Exception e) {
                System.out.println(e);
                System.out.println("Caught Error while allocating x in LinearRegressor");
                return;
            }

            try {
            this.y = y;
            } catch (Exception e) {
                System.out.println(e);
                System.out.println("Caught Error while initializing y in LinearRegressor");
                return;
            }

            try {
            this.lr = lr;
            } catch (Exception e) {
                System.out.println(e);
                System.out.println("Caught Error while initializing learning rate in LinearRegressor");
                return;
            }
            this.y_pred = new double[y.length];
        }
    
        // Simple squared error as the loss function
        public double squaredError() {
            this.loss = 0; // Reset loss
            for (int i = 0; i < this.x.length; i++) {
                this.loss += Math.pow((this.y[i] - this.y_pred[i]), 2);
            }
            return this.loss;
        }
    
        // Compute gradient and update weights
        public void simpleGradient() {
            this.grad = new double[this.w.length]; // Reset gradient
            double[] partialDerivativeSum = new double[this.w.length]; // For the formula Sigma(i=1 to n) (y_actual - y_pred) * x, for the partial derivative of SE
            for (int i = 0; i < x.length; i++) {
                for (int j = 0; j < this.w.length; j++)
                    partialDerivativeSum[j] += (y[i] - y_pred[i]) * mean(x[i]);
            }
            for (int i = 0; i < this.w.length; i++)
                this.grad[i] = -1.0 / x.length * partialDerivativeSum[i];
    
            for (int i = 0; i < this.w.length; i++)
                this.w[i] = this.w[i] - (this.lr * this.grad[i]);
        }

        // Function to compute mean for the gradient
        public double mean(double[] x){
            double sum = 0.0;
            for(int i = 0; i < x.length; i++)
                sum += x[i];
            return sum / x.length;
        }
    
        // Function for forward pass
        public void forwardPass() {
            for (int i = 0; i < x.length; i++)
                this.y_pred[i] = 0.0;
            for (int i = 0; i < x.length; i++) {
                for (int j = 0; j < this.x[i].length; j++)
                    this.y_pred[i] += this.w[j] * this.x[i][j];
                this.y_pred[i] += this.b;
            }
        }

        // Function to predict
        public double[] predict(double[][] x_t) {
            double[] y_t = new double[x_t.length];
            for (int i = 0; i < x_t.length; i++) {
                y_t[i] = 0.0;
                for (int j = 0; j < x_t[i].length; j++)
                    y_t[i] += this.w[j] * x_t[i][j];
                y_t[i] += this.b;
            }
            return y_t;
        }

        // Function to fetch non-static values
        public double[] fetchWeights(){
            return this.w;
        }

}
