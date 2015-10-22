//Ejemplo CNN
//http://semantive.com/deep-learning-examples/
//Ejemplo de entrada de imagenes
//http://deeplearning4j.org/image-data-pipeline.html
//Documentacion DL4J
//http://deeplearning4j.org/doc/allclasses-noframe.html

package com.escom.CNNS;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.escom.UTIL.Utileria;

public class Redes {

	private static final Logger LOG = LoggerFactory.getLogger(Redes.class);


	public static MultiLayerConfiguration  getConfiguration1() {
		LOG.info("Construyendo red neuronal 1");
		MultiLayerConfiguration.Builder  builder = new NeuralNetConfiguration.Builder()
		.seed(Utileria.SEMILLA)
		.batchSize(Utileria.TAM_LOTE)
		.iterations(Utileria.ITERACIONES)
		.momentum(0.9)
		.regularization(true)
		.l1(1e-1).l2(2e-4)
		.useDropConnect(true)
		.constrainGradientToUnitNorm(true)
		.miniBatch(true)
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
			.nOut(Utileria.ETIQUETAS.size())
			.dropOut(0.5)
			.weightInit(WeightInit.VI)
			.build())
		.inputPreProcessor(0, new FeedForwardToCnnPreProcessor(Utileria.WIDTH, Utileria.HEIGHT))
		.inputPreProcessor(4, new CnnToFeedForwardPreProcessor(Utileria.WIDTH, Utileria.HEIGHT))
		.backprop(true).pretrain(false);

		new ConvolutionLayerSetup(builder, Utileria.HEIGHT, Utileria.WIDTH, Utileria.N_CANALES);

		return builder.build();
	}

}

