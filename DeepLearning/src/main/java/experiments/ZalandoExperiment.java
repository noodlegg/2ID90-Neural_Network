package experiments;

import java.io.IOException;
import nl.tue.s2id90.dl.NN.Model;
import nl.tue.s2id90.dl.NN.initializer.Gaussian;
import nl.tue.s2id90.dl.NN.layer.Flatten;
import nl.tue.s2id90.dl.NN.layer.InputLayer;
import nl.tue.s2id90.dl.NN.layer.OutputSoftmax;
import nl.tue.s2id90.dl.NN.loss.CrossEntropy;
import nl.tue.s2id90.dl.NN.optimizer.Optimizer;
import nl.tue.s2id90.dl.NN.optimizer.SGD;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import nl.tue.s2id90.dl.NN.validate.Classification;
import nl.tue.s2id90.dl.experiment.GUIExperiment;
import nl.tue.s2id90.dl.input.InputReader;
import nl.tue.s2id90.dl.input.MNISTReader;
import nl.tue.s2id90.dl.javafx.FXGUI;
import nl.tue.s2id90.dl.javafx.ShowCase;

public class ZalandoExperiment extends GUIExperiment {
    // (hyper)parameters
    int batchSize = 68;
    int epochs = 5;                // number of epochs a training takes
    double learningRate = 0.15;    // parameter for gradient descent optimization method

    // add two fields to your Experiment class
    String[] labels = {
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
        "Shirt", "Sneaker", "Bag", "Ankle boot"
    };
    ShowCase showCase = new ShowCase(i -> labels[i]);

    public void go() throws IOException {
        // you are going to add code here
        // read input and print some information on the data
        InputReader reader = MNISTReader.fashion(batchSize);
        System.out.println("Reader info:\n" + reader.toString());

        // print a record
        reader.getValidationData(1).forEach(System.out::println);
        
        FXGUI.getSingleton().addTab("show case", showCase.getNode());
        showCase.setItems(reader.getValidationData(100));

        int inputs = reader.getInputShape().getNeuronCount();
        int outputs = reader.getOutputShape().getNeuronCount();
        Model model = createModel(inputs, outputs);
        System.out.println(model);  // prints summary of the model
        model.initialize(new Gaussian());   // initializes model weights

        // Training: create and configure SGD && train model
        Optimizer sgd = SGD.builder()
                .model(model)
                .learningRate(learningRate)
                .validator(new Classification())
                .build();
        trainModel(sgd, reader, epochs, 0);

    }

    Model createModel(int inputs, int outputs) {
        Model model = new Model(new InputLayer("In", new TensorShape(28,28,1), true));
        model.addLayer(new Flatten("Flatten", new TensorShape(28,28,1)));
        model.addLayer(new OutputSoftmax("Out", new TensorShape(inputs), outputs, new CrossEntropy()));
        return model;
    }
    
    public void onEpochFinished(Optimizer sgd, int epoch) {
        super.onEpochFinished(sgd, epoch);
        showCase.update(sgd.getModel());
    }

    public static void main(String[] args) throws IOException {
        new ZalandoExperiment().go();
    }
}
