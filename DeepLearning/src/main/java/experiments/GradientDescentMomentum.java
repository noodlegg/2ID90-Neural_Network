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
    INDArray v;
    INDArray prevV;
    
    // Does a gradient descent step with factor 'minus learningRate' and  corrected for batchSize.
    @Override public void update(INDArray array, boolean isBias, double learningRate, 
            int batchSize, INDArray gradient) {
        double factor = -(learningRate/batchSize);
        
        // on first call of this method, create update vector
        if (update == null) {
            update = gradient.dup('f').assign(0);
        }
        
        // v_prev = v;
        prevV = v;
        // v = mu * v - learningRate * dx
        v = v.mul(mu).add(gradient.mul(factor));
        // -mu * v_prev + (1 + mu)
        prevV = prevV.mul(-mu).add((1+mu));
        // x <-- x + (-mu * v_prev + (1 + mu)) * v
        update = update.add(prevV).mul(v);
        
        // array <-- array + factor * gradient
        Nd4j.getBlasWrapper().level1().axpy(array.length(), 1, update, array);
    }
}
