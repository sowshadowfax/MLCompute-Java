package MLCompute;
import java.util.Random;

public class MultiLinearRegressor {
    
        // Declaring variables for the equation y = w1x1 + w2x1 + ... + wnx1 + b
        double[] w = {};
        double[] x = {};
        double b = 0.0;
        double[] y = {};
        double[] y_pred = {};
    
        // Declaring variables for gradient descent / back-propagation
        double[] grad = {};
        double lr = 0.01; // Learning rate
        double loss = 0.0;
    
        // Initializes the weights randomly and assigns bias
        public void initialize(double b, double[] x, double[] y, double lr) {
            Random rand = new Random();
            this.w = new double[x.length];
            for (int i = 0; i < this.w.length; i++)
                this.w[i] = rand.nextDouble();
            this.b = b;
            this.x = x;
            this.y = y;
            this.lr = lr;
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
                    partialDerivativeSum[j] += (y[i] - y_pred[i]) * x[i];
            }
            for (int i = 0; i < this.w.length; i++)
                this.grad[i] = -1.0 / x.length * partialDerivativeSum[i];
    
            for (int i = 0; i < this.w.length; i++)
                this.w[i] = this.w[i] - (this.lr * this.grad[i]);
        }
    
        // Function for forward pass
        public void forwardPass() {
            for (int i = 0; i < x.length; i++)
                this.y_pred[i] = 0.0;
            for (int i = 0; i < x.length; i++) {
                for (int j = 0; j < this.w.length; j++)
                    this.y_pred[i] += this.w[j] * this.x[i];
                this.y_pred[i] += this.b;
            }
        }

        // Function to predict
        public double[] predict(double[] x_t) {
            double[] y_t = new double[x_t.length];
            for (int i = 0; i < x_t.length; i++) {
                y_t[i] = 0.0;
                for (int j = 0; j < this.w.length; j++)
                    y_t[i] += this.w[j] * x_t[i];
                y_t[i] += this.b;
            }
            return y_t;
        }

        // Function to fetch non-static values
        public double[] fetchWeights(){
            return this.w;
        }

}
