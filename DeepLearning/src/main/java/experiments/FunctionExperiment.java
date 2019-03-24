package experiments;

import nl.tue.s2id90.dl.experiment.Experiment;
import java.io.IOException;
import nl.tue.s2id90.dl.NN.Model;
import nl.tue.s2id90.dl.NN.layer.InputLayer;
import nl.tue.s2id90.dl.NN.layer.SimpleOutput;
import nl.tue.s2id90.dl.NN.loss.MSE;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import nl.tue.s2id90.dl.input.GenerateFunctionData;
import nl.tue.s2id90.dl.input.InputReader;

public class FunctionExperiment extends Experiment {
    // (hyper)parameters
    int batchSize = 32;

    public void go() throws IOException {
        // you are going to add code here
        // read input and print some information on the data
        InputReader reader = GenerateFunctionData.THREE_VALUED_FUNCTION(batchSize);
        System.out.println("Reader info:\n" + reader.toString());

        int inputs = reader.getInputShape().getNeuronCount();
        int outputs = reader.getOutputShape().getNeuronCount();
        Model model = createModel(inputs, outputs);
        System.out.println(model);

    }

    Model createModel(int inputs, int outputs) {
        Model model = new Model(new InputLayer("In", new TensorShape(inputs), true));
        model.addLayer(new SimpleOutput("Out", new TensorShape(inputs), outputs, new MSE(), true));
        return model;
    }

    public static void main(String[] args) throws IOException {
        new FunctionExperiment().go();
    }
}
