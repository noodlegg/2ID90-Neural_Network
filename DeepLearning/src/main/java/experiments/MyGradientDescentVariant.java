/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experiments;

import nl.tue.s2id90.dl.NN.optimizer.Optimizer;
import nl.tue.s2id90.dl.NN.optimizer.SGD;
import nl.tue.s2id90.dl.NN.optimizer.update.UpdateFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author dennis
 */
public class MyGradientDescentVariant implements UpdateFunction {
    double v = 0;
    int momentum = 0; //put a value for it.
    INDArray update;
    @Override
    public void update(INDArray value, boolean isBias, 
            double learningRate, int batchSize, INDArray gradient) {
        if(update == null){
            update = gradient.dup('f').assign(0);
        }
        v = (momentum * v) -(learningRate/batchSize);
        Nd4j.getBlasWrapper().level1().axpy( value.length(), v , gradient, value );
                                            // value <-- value + factor * gradient
    }
    SGD.SGDBuilder sgd = SGD.builder();
}
