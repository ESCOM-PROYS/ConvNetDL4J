package org.deeplearning4j.deeplearning4j_examples;


import org.canova.api.io.WritableConverter;
import org.canova.api.io.converters.WritableConverterException;
import org.canova.api.io.data.IntWritable;
import org.canova.api.io.data.Text;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.records.reader.impl.ComposableRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.writable.Writable;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.*;

/**
 * Created by willow on 5/11/15.
 */
public class CNN_EX {

    private static final Logger log = LoggerFactory.getLogger(CNN_EX.class);
    private static final List<String> ETIQUETAS = Arrays.asList("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck");
    private static int numFilas = 32;
    private static int numColumnas = 32;
    private static int nChannels = 1;
    private static int outputNum = 10;
    private static int numSamples = 2000;
    private static int batchSize = 500;
    private static int iterations = 10;
    private static int splitTrainNum = (int) (batchSize*.8);
    private static int seed = 123;
    private static int listenerFreq = iterations/5;
    private static DataSet mnist;
    private static SplitTestAndTrain trainTest;
    private static DataSet trainInput;
    private static List<INDArray> testInput = new ArrayList<>();
    private static List<INDArray> testLabels = new ArrayList<>();

    public static void main(String[] args) throws Exception {
        
        log.info("Load data....");
        //DataSetIterator dsetIterator = new MnistDataSetIterator(batchSize,numSamples, true);
        RecordReader recordReader = loadData();

        // Se sobreescribe convert para sustituir la etiqueta de la imagen por su indice en la lista de etiquetas:  frog -> 6
        DataSetIterator dsetIterator = new RecordReaderDataSetIterator(recordReader, new WritableConverter() {
            @Override
            public Writable convert(Writable writable) throws WritableConverterException {
                if (writable instanceof Text) {
              	  //System.out.println("Writable: " + writable.toString());
                    String label = writable.toString().replaceAll("\u0000", "");
                    int index = ETIQUETAS.indexOf(label);
                    //System.out.println(++contadorBatch + "Label: " + label + " index: " + index);
                    return new IntWritable(index);
                }else{
              	  System.out.println("ERROR: No es una instancia de Text. [dataSetIterator]");
                }
                return writable;
            }
        }, batchSize, 1024, 10);

        
        log.info("CONSTRUYENDO MODELO....");
        
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .batchSize(batchSize)
                .iterations(iterations)
                .constrainGradientToUnitNorm(true)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list(3)
                .layer(0, new ConvolutionLayer.Builder(10, 10)
                        .nIn(nChannels)
                        .nOut(6).stride(new int[]{1, 1})
                        .weightInit(WeightInit.VI)
                        .activation("relu")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {2,2})
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .weightInit(WeightInit.VI)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false);

        new ConvolutionLayerSetup(builder,numFilas,numColumnas,nChannels);

        MultiLayerConfiguration conf = builder.build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        
        log.info("Train model....");
        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));
        while(dsetIterator.hasNext()) {
            mnist = dsetIterator.next();
            trainTest = mnist.splitTestAndTrain(splitTrainNum, new Random(seed)); // train set that is the result
            trainInput = trainTest.getTrain(); // get feature matrix and labels for training
            testInput.add(trainTest.getTest().getFeatureMatrix());
            testLabels.add(trainTest.getTest().getLabels());
            model.fit(trainInput);
        }

        log.info("Evaluate weights....");

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum);
        for(int i = 0; i < testInput.size(); i++) {
            INDArray output = model.output(testInput.get(i));
            eval.eval(testLabels.get(i), output);
        }
        INDArray output = model.output(testInput.get(0));
        eval.eval(testLabels.get(0), output);
        log.info(eval.stats());
        log.info("****************Example finished********************");


    }
    
    
    /**
     * @return RecordReader
     * @throws Exception
     */
    public static RecordReader loadData() throws Exception {
          System.out.println("Cargando imagenes desde: " + System.getProperty("user.home") + "/deepLearning/sets/cifar10/img/train/" );
          RecordReader imageReader = new ImageRecordReader(numFilas, numColumnas, false);
          imageReader.initialize(new FileSplit(new File(System.getProperty("user.home") + "/deepLearning/sets/cifar10/img/train/")));
         
          System.out.println("Cargando archivo de etiquetas...");
          RecordReader labelsReader = new CSVRecordReader();
          labelsReader.initialize(new FileSplit(new File(System.getProperty("user.home") + "/deepLearning/sets/cifar10/trainLabelsUN.csv")));
          /*
          * Eliminar etiquetas de numeración de la colección:
          int cont =0;
          Collection<Writable> label;
          try{
          while((label=labelsReader.next()) != null){
        	 label.remove(label.toArray()[0]);
        	 //System.out.println("Tam elementos: " + label.size() + " elem: " + label.toArray()[0]);
          }
          }catch(Exception e){}
          */
          return new ComposableRecordReader(imageReader, labelsReader);
      }
    
}