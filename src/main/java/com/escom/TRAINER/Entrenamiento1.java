package com.escom.TRAINER;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.canova.api.io.WritableConverter;
import org.canova.api.io.converters.WritableConverterException;
import org.canova.api.io.data.IntWritable;
import org.canova.api.io.data.Text;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.writable.Writable;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;

import com.escom.CNNS.Redes;
import com.escom.LOADER.LoaderCsv;
import com.escom.UTIL.Utileria;

public class Entrenamiento1 {

	/**
	 * Objeto para imprimir los mensajes de tipo Logger 
	 */
	public static final org.slf4j.Logger LOG = org.slf4j.LoggerFactory.getLogger(Entrenamiento1.class);

	public static void main(String args[]) throws Exception{

		DataSet cifarDataSet;
		SplitTestAndTrain trainAndTest;
		DataSet trainInput;
		List<INDArray> testInput = new ArrayList<>();
		List<INDArray> testLabels = new ArrayList<>();

		Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

		//MultiLayerNetwork model = new MultiLayerNetwork(Redes.getConfiguration1());
		MultiLayerNetwork model = new MultiLayerNetwork(Redes.getConfiguration2());
		model.init();
		LOG.info("Configuración del modelo:");
		model.printConfiguration();

		DataSetIterator dataSetIterator;
		try {
			dataSetIterator = LoaderCsv.loadData1();
		} catch (IOException | InterruptedException e) {
			LOG.error("Error al cargar los datos de entrenamiento");
			throw e;
		}

		LOG.info("\n******* ENTRENANDO RED *******\n");

		while (dataSetIterator.hasNext()) {
			model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(Utileria.LISTENER_FREQ)));
			cifarDataSet = dataSetIterator.next();
			trainAndTest = cifarDataSet.splitTestAndTrain(Utileria.SPLIT_TRAIN_NUM, new Random(Utileria.SEMILLA));//Definir un subconjunto del batch de tamanio "splitTrainNum" para entrenar.
			
			trainInput = trainAndTest.getTrain(); 
			testInput.add(trainAndTest.getTest().getFeatureMatrix());
			testLabels.add(trainAndTest.getTest().getLabels());
			
			model.fit(trainInput);

		}


		LOG.info("\n\n******* EVALUANDO RED *******\n");
		Evaluation eval = new Evaluation(Utileria.ETIQUETAS.size());
		for (int i = 0; i < testInput.size(); i++) {
			INDArray output = model.output(testInput.get(i));
			eval.eval(testLabels.get(i), output);
			LOG.info(eval.stats());
		}
	}



	public static void imprimirLoteIMG(DataSetIterator set){
		while(set.hasNext()){
			DataSet datS = set.next();
			List<DataSet> lista = datS.asList();
			System.out.println("Tamanio lista:" + lista.size());
			for (DataSet ds : lista){
				List<DataSet> listaDS = ds.get(0).asList();
				System.out.println("Tamanio " + listaDS.size());
				System.out.println(listaDS);
			}
		}
	} 


	public static void imprimirPesosCapa(MultiLayerNetwork modelo){
		LOG.info("Mostrando pesos de la red...");
		for(Layer capa : modelo.getLayers()){
			INDArray pesos = capa.getParam(DefaultParamInitializer.WEIGHT_KEY);
			LOG.info("PESOS DE LA CAPA " + capa.getIndex() + ": \n" + pesos);
		}
	}


}
