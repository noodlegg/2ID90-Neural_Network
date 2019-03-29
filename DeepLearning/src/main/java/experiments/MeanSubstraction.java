package experiments;
import java.util.List;
import nl.tue.s2id90.dl.NN.tensor.Tensor;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import nl.tue.s2id90.dl.NN.transform.DataTransform;
import org.nd4j.linalg.api.ndarray.INDArray;

public class MeanSubstraction implements DataTransform {
    Double mean;
    @Override public void fit(List<TensorPair> data) {
        if (data.isEmpty()) {
            throw new IllegalArgumentException("Empty dataset");
        }
        for (TensorPair pair : data) {
            //..       
            Tensor t = pair.model_input;
            INDArray a = t.getValues();
        }
        //...
    }
    @Override public void transform(List<TensorPair> data) {
        //To do
    }
}