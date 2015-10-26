package com.escom.UTIL;

import java.awt.FileDialog;
import java.io.DataOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.ResourceBundle;

import javax.swing.JFileChooser;
import javax.swing.JFrame;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.factory.Nd4j;

public class Utileria {
	/**
	 * Objeto para imprimir los mensajes de tipo Logger 
	 */
	public static final org.slf4j.Logger LOG = org.slf4j.LoggerFactory.getLogger(Utileria.class);
	
	/**
	 * Variables de configuracion
	 */
	public static int WIDTH;			//Tamanio Ancho de la imagen
	public static int HEIGHT;			//Tamanio Alto de la imagen
	public static int N_CANALES;		//Numero de canales de la imagen
	public static int TAM_LOTE;
	public static int ITERACIONES;
	public static int SPLIT_TRAIN_NUM;
	public static int LISTENER_FREQ;
	public static int SEMILLA;
	public static List<String> ETIQUETAS;
	
	public static String HOME_DIR;		//Directorio Home del del usuario
	public static String SEP;			//Caracter de separacion del sistema (/ = Linux, \\ = Windows, etc..)
	public static String DIR_IMGS_ENTR;		//Path del directorio de imagens
	public static String DIR_LABELS;	//Path del archivo que contiene los labels
	/**
	 * Objeto para obeter las propiedades del archivo cfg.properties 
	 * que se encuetra en resorce
	 */
	private static ResourceBundle resource;
	
	/**
	 * Codigo estatico para inicializar todas las variables estaticas
	 * de la clase
	 */
	static{
		
		try {
			resource = ResourceBundle.getBundle("cfg");	
			
			WIDTH = getIntPropiedad("WIDTH");
			LOG.info("Valor de WIDTH: "+WIDTH);
			HEIGHT = getIntPropiedad("HEIGHT");
			LOG.info("Valor de HEIGHT: "+HEIGHT);
			TAM_LOTE = getIntPropiedad("TAM_LOTE");
			LOG.info("Valor de TAM_LOTE: "+TAM_LOTE);
			ITERACIONES = getIntPropiedad("ITERACIONES");
			LOG.info("Valor de ITERACIONES: "+ITERACIONES);
			N_CANALES = getIntPropiedad("N_CANALES");
			LOG.info("Valor de N_CANALES: "+N_CANALES);
			SEMILLA = getIntPropiedad("SEMILLA");
			LOG.info("Valor de SEMILLA: "+SEMILLA);
			ETIQUETAS = Arrays.asList(getStringPropiedad("ETIQUETAS").split(","));
			LOG.info("Etiqutas: "+ETIQUETAS);
			SPLIT_TRAIN_NUM = getIntPropiedad("SPLIT_ENTRENAMIENTO");
			LOG.info("SPLIT_TRAIN_NUM: "+SPLIT_TRAIN_NUM);
			LISTENER_FREQ = getIntPropiedad("FRECUENCIA");
			LOG.info("LISTENER_FREQ: "+LISTENER_FREQ);
			
			HOME_DIR = System.getProperty("user.home");
			LOG.info("Directorio Home: "+HOME_DIR);
			SEP = System.getProperty("file.separator");
			LOG.info("Separador de archuvos: "+SEP);
			DIR_IMGS_ENTR = HOME_DIR
					+ SEP
					+ getStringPropiedad("NOM_DIRECTORIO_ENTRENAMIENTO")
					+ SEP
					+ getStringPropiedad("NOM_DIRECTORIO_IMGS_ENTRENAMIENTO");
			LOG.info("Path del directorio de imagenes de entrenamiento: "+DIR_IMGS_ENTR);
			DIR_LABELS = HOME_DIR
					+ SEP
					+ getStringPropiedad("NOM_DIRECTORIO_ENTRENAMIENTO")
					+ SEP
					+ getStringPropiedad("NOM_ARCHIVO_ETIQUETAS");
			LOG.info("Path del archivo con las etiquetas: "+DIR_LABELS);
		} catch (Exception e) {
			LOG.error("Error al inicializar las varialbes estaticas");
			e.printStackTrace();
		}
	}
	
	/**
	 * Obtiene el valor que tiene el paramento keyPropiedad en el archivo cfg.properties 
	 * @param keyPropiedad
	 * @return El valor de la propiedad en tipo cadena
	 * @throws Exception
	 */
	public static String getStringPropiedad(String keyPropiedad)throws Exception{
		try{
			return resource.getString(keyPropiedad);
		}catch(Exception ex){
			LOG.error("Error inesperado: "+ex.getMessage());
			throw ex;
		}
	}
	
	/**
	 * Obtiene el valor que tiene el paramento keyPropiedad en el archivo cfg.properties
	 * y la trata de convertir a entero
	 * @param keyPropiedad
	 * @return El valor de la propiead en tipo entero
	 * @throws Exception
	 */
	public static int getIntPropiedad(String keyPropiedad)throws Exception{
		try{
			
			return Integer.parseInt(resource.getString(keyPropiedad));
			
		}catch(NumberFormatException nfe){
			LOG.error("Error al obtener la propiedad "+keyPropiedad
					+" no es un entero tiene un valor de "
					+resource.getString(keyPropiedad));
			throw nfe;
		}catch(Exception ex){
			LOG.error("Error inesperado: "+ex.getMessage());
			throw ex;
		}
	}
	
	/**
	 * Guarda una carpeta con dos archivos de configuracion de la red nerunal
	 * el primero guarda los pesos de las neuronas de la red neuronal en un formato .bin, y el 
	 * segundo guarda la estructura de lared neuronal en un formato .json
	 * @param model
	 * @throws IOException 
	 */
	public static void guardarRedNeuronal(MultiLayerNetwork model) throws IOException{
		DataOutputStream dos = null;
		OutputStream fos= null;
		JFileChooser jfc= new JFileChooser();
		JFrame jf = new JFrame();
		try{
			LOG.info("Guardando los pesos de las redes neuronales...");
			jfc.showOpenDialog(jf);
			fos = Files.newOutputStream(Paths.get("coefficients.bin"));
		    dos = new DataOutputStream(fos);
		    dos.flush();
		    Nd4j.write(model.params(), dos);
		    LOG.info("Guardando los pesos de las redes neuronales <ok>");
		    LOG.info("Guardando la arquitectura de la red neuronal...");
		    FileUtils.write(new File("conf.json"), model.getLayerWiseConfigurations().toJson());
		    LOG.info("Guardando la arquitectura de la red neuronal <ok>");
		}catch(IOException ioe){
			LOG.error("Error al guardar la configuración de la red neuronal: "+ioe.getMessage());
			throw ioe;
		}
		finally{
			try{
				if(dos != null){
				    dos.close();	
				}
				if(fos != null){
					fos.close();
				}	
			}catch(Exception e){
				LOG.error("Error inesperado: ");
				e.printStackTrace();
			}
		}
	    
//	    MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File("conf.json")));
//	    DataInputStream dis = new DataInputStream(new FileInputStream("coefficients.bin"));
//	    INDArray newParams = Nd4j.read(dis);
//	    dis.close();
//	    MultiLayerNetwork savedNetwork = new MultiLayerNetwork(confFromJson);
//	    savedNetwork.init();
//	    savedNetwork.setParameters(newParams);
//	    System.out.println("Original network params " + model.params());
//	    System.out.println(savedNetwork.params());
	}
	
	public static MultiLayerNetwork cargarRedNueronal(){
		return null;
	}

}
