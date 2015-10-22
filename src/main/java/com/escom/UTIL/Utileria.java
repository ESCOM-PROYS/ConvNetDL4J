package com.escom.UTIL;

import java.util.Arrays;
import java.util.List;
import java.util.ResourceBundle;

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
					+ getStringPropiedad("NOM_DIRECTPRIO_ENTRENAMIENTO")
					+ SEP
					+ getStringPropiedad("NOM_DIRECTORIO_IMGS_ENTRENAMIENTO");
			LOG.info("Path del directorio de imagenes de entrenamiento: "+DIR_IMGS_ENTR);
			DIR_LABELS = HOME_DIR
					+ SEP
					+ getStringPropiedad("NOM_DIRECTPRIO_ENTRENAMIENTO")
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

}
