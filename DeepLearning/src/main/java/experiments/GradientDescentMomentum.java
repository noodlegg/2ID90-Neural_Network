package experiments;

import nl.tue.s2id90.dl.NN.optimizer.update.UpdateFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Thomas
 */
public class GradientDescentMomentum implements UpdateFunction {
    INDArray update;
    double mu = 0.9;
    INDArray prevV;
    
    // Does a gradient descent step with factor 'minus learningRate' and  corrected for batchSize.
    @Override public void update(INDArray array, boolean isBias, double learningRate, 
            int batchSize, INDArray gradient) {
        double factor = -(learningRate/batchSize);
        
        // on first call of this method, create update vector
        if (update == null) {
            update = gradient.dup('f').assign(0);
        }
        
        // v = mu * v - learningRate * dx
        update = update.mul(mu).add(gradient.mul(factor));

        // x <-- x + v
        // array <-- array + factor * gradient
        Nd4j.getBlasWrapper().level1().axpy(array.length(), 1, update, array);
    }
}
