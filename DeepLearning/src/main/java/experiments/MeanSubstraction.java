package experiments;
import java.util.List;
import nl.tue.s2id90.dl.NN.tensor.Tensor;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import nl.tue.s2id90.dl.NN.transform.DataTransform;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class MeanSubstraction implements DataTransform {
    Double mean = 0.0;
    Double sum = 0.0;    // sum of all INDArray values
    int numOfINDArrayElements = 0; // total amount of INDArray elements
    @Override public void fit(List<TensorPair> data) {
        if (data.isEmpty()) {
            throw new IllegalArgumentException("Empty dataset");
        }
        for (TensorPair pair : data) {
            //..       
            Tensor t = pair.model_input;
            // Convert the INDArray into a single flat array
            INDArray a = t.getValues();
            a = Nd4j.toFlattened(a);
            sum += a.sumNumber().doubleValue();
            numOfINDArrayElements += a.length();
        }
        mean = sum / numOfINDArrayElements;
    }
    @Override public void transform(List<TensorPair> data) {
        for (TensorPair pair : data) {
            Tensor t = pair.model_input;
            INDArray a = t.getValues();
            a.subi(mean);
        }
    }
}