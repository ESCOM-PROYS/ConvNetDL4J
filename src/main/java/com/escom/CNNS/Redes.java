//Ejemplo CNN
//http://semantive.com/deep-learning-examples/
//Ejemplo de entrada de imagenes
//http://deeplearning4j.org/image-data-pipeline.html
//Documentacion DL4J
//http://deeplearning4j.org/doc/allclasses-noframe.html
//http://deeplearning4j.org/multinetwork.html

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
		.learningRate(0.001) // Rango [0.001 , 0.1] Menor valor -> mayor tiempo de entrenamiento y resultados más precisos
		.momentum(0.9) // parametro útil para la convergencia. Afecta el grado de cambio en los pesos durante el aprendizaje. Momento grande aumenta velocidad. Valores [0-1]
		.l1(1e-1)
		.l2(2e-4)
		.regularization(true)
		.useDropConnect(true)//Permite "limitar" la conexión entre las neuronas de las capas haciendo cero algunas activaciones aleatoriamente.
		.constrainGradientToUnitNorm(true)
		.miniBatch(true)
		.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
		.list(6)
		.layer(0, new ConvolutionLayer.Builder(new int[]{6, 6}) // KERNEL DE 6X6 .....   Dimension espacial de salida: (W−F+2P)/S+1
			.nIn(Utileria.N_CANALES)
			.nOut(20)
			.stride(new int[]{1, 1})
			.padding(new int[]{0, 0})
			.activation("relu")// f(x) = Max(0, x) , para agregar No-linealidad
			.weightInit(WeightInit.XAVIER) //Determina automaticamente la escala de inicialización basado en el numero de neuronas de entrada y salida. Definiciones en WeightInit.class 
			.build())
		.layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2}) //kernel 2x2
			.build())
		.layer(2, new ConvolutionLayer.Builder(new int[]{6, 6}) // kernel 6x6
			.nIn(20)
			.nOut(40)
			.stride(new int[]{1, 1})
			.activation("relu")
			.weightInit(WeightInit.XAVIER)
			.build())
		.layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
			.build())
		.layer(4, new DenseLayer.Builder()
			.nIn(40 * 5 * 5)
			.nOut(1000)
			.activation("relu")
			.weightInit(WeightInit.XAVIER)
			.dropOut(0.5)//Para prevenir Overfitting. 
			.build())
		.layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MSE) //MSE: Mean Squared Error: Linear Regression
			.nIn(1000)
			.nOut(Utileria.ETIQUETAS.size())
			.dropOut(0.5)//
			.weightInit(WeightInit.XAVIER)
			.activation("softmax")
			.build())
		.inputPreProcessor(4, new CnnToFeedForwardPreProcessor(Utileria.WIDTH, Utileria.HEIGHT))
		.inputPreProcessor(0, new FeedForwardToCnnPreProcessor(Utileria.WIDTH, Utileria.HEIGHT))
		.backprop(true).pretrain(false);

		new ConvolutionLayerSetup(builder, Utileria.HEIGHT, Utileria.WIDTH, Utileria.N_CANALES);

		return builder.build();
	}

	
	
	public static MultiLayerConfiguration getConfiguration2(){
		
		
		MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(Utileria.SEMILLA)
                .batchSize(500)
                .iterations(10)
                .constrainGradientToUnitNorm(true)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list(3)
                .layer(0, new ConvolutionLayer.Builder(10, 10)
                        .nIn(3)
                        .nOut(6).stride(new int[]{1, 1})
                        .weightInit(WeightInit.VI)
                        .activation("relu")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {2,2})
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(10)
                        .weightInit(WeightInit.VI)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false);

        new ConvolutionLayerSetup(builder,32,32,1);

        return builder.build();
		
	}
	
	
	
	
}

