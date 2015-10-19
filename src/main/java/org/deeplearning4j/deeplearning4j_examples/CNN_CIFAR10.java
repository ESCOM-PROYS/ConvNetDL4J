//Ejemplo CNN
//http://semantive.com/deep-learning-examples/
//Ejemplo de entrada de imagenes
//http://deeplearning4j.org/image-data-pipeline.html
//Documentacion DL4J
//http://deeplearning4j.org/doc/allclasses-noframe.html

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
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Random;

public class CNN_CIFAR10 {
	
  private static final int WIDTH = 32;
  private static final int HEIGHT = 32;
  private static final int TAM_BATCH = 500;
  private static final int ITERACIONES = 10;
  private static final int SEED = 123;
  private static final Logger log = LoggerFactory.getLogger(CNN_CIFAR10.class);
  private static final List<String> ETIQUETAS = Arrays.asList("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck");

  
  public static void main(String[] args) throws Exception {

      int splitTrainNum = (int) (TAM_BATCH * 0.8);
      int listenerFreq = ITERACIONES / 5;

      DataSet cifarDataSet;
      SplitTestAndTrain trainAndTest;
      DataSet trainInput;
      List<INDArray> testInput = new ArrayList<>();
      List<INDArray> testLabels = new ArrayList<>();

      Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

      RecordReader recordReader = loadData();

      // Se sobreescribe convert para sustituir la etiqueta de la imagen por su indice en la lista de etiquetas:  frog -> 6
      DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader, new WritableConverter() {
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
      }, TAM_BATCH, 1024, 10);

      //System.out.println("Tamanio datasetIt = " + dataSetIterator.totalExamples());
      
      MultiLayerNetwork model = new MultiLayerNetwork(getConfiguration());
      model.init();
      System.out.println("\n\nConfiguración del modelo:");
      model.printConfiguration();
      System.out.println();
      
      log.info("\n******* ENTRENANDO RED *******\n");
      
      if(!dataSetIterator.hasNext()){
    	  System.err.println("ERROR: conjunto de datos de entrenamiento vacío...");
      } 
      else{
    	  while (dataSetIterator.hasNext()) {
    		  int contador = 0;
        	  //System.out.println("Cont-" + ++contador);
              model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));
              cifarDataSet = dataSetIterator.next();
              trainAndTest = cifarDataSet.splitTestAndTrain(splitTrainNum, new Random(SEED));//Definir un subconjunto del batch de tamanio "splitTrainNum" para entrenar. 
              trainInput = trainAndTest.getTrain(); 	// D
              testInput.add(trainAndTest.getTest().getFeatureMatrix());
              testLabels.add(trainAndTest.getTest().getLabels());
              model.fit(trainInput);
              //model.fit(cifarDataSet);
          }

          log.info("\n\n******* EVALUANDO RED *******\n");
          Evaluation eval = new Evaluation(ETIQUETAS.size());
          for (int i = 0; i < testInput.size(); i++) {
              INDArray output = model.output(testInput.get(i));
              eval.eval(testLabels.get(i), output);
          }
          log.info(eval.stats());
      }
      
  }

  
  
  public static MultiLayerConfiguration getConfiguration() {
      MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
              .seed(SEED)
              .batchSize(TAM_BATCH)
              .iterations(ITERACIONES)
              .momentum(0.9)
              .regularization(true)
              .constrainGradientToUnitNorm(true)
              .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
              .list(6)
              .layer(0, new ConvolutionLayer.Builder(new int[]{5, 5}) // KERNEL DE 5X5
                      .nIn(1)
                      .nOut(20)
                      .stride(new int[]{1, 1})
                      .activation("relu")// f(x) = Max(0, x) , para agregar No-linealidad
                      .weightInit(WeightInit.VI) //VI: Sample weights from variance normalized initialization (Glorot) -> definiciones en WeightInit.class 
                      .build())
              .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2}) //kernel 2x2
                      .build())
              .layer(2, new ConvolutionLayer.Builder(new int[]{5, 5}) // kernel 5x5
                      .nIn(20)
                      .nOut(40)
                      .stride(new int[]{1, 1})
                      .activation("relu")
                      .weightInit(WeightInit.VI)
                      .build())
              .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                      .build())
              .layer(4, new DenseLayer.Builder()
                      .nIn(40 * 5 * 5)
                      .nOut(1000)
                      .activation("relu")
                      .weightInit(WeightInit.VI)
                      .dropOut(0.5)
                      .build())
              .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MSE) //MSE: Mean Squared Error: Linear Regression
                      .nIn(1000)
                      .nOut(ETIQUETAS.size())
                      .dropOut(0.5)
                      .weightInit(WeightInit.VI)
                      .build())
              .inputPreProcessor(0, new FeedForwardToCnnPreProcessor(WIDTH, HEIGHT))
              .inputPreProcessor(4, new CnnToFeedForwardPreProcessor(WIDTH,HEIGHT))
              .backprop(true).pretrain(false)
              .build();

      return conf;
  }


  /**
 * @return RecordReader
 * @throws Exception
 */
public static RecordReader loadData() throws Exception {
      System.out.println("Cargando imagenes desde: " + System.getProperty("user.home") + "/deepLearning/sets/cifar10/img/train/" );
      RecordReader imageReader = new ImageRecordReader(WIDTH, HEIGHT, false);
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

