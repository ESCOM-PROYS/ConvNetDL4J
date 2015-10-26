package com.escom.UTIL;

import java.awt.FileDialog;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.ResourceBundle;

import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.filechooser.FileFilter;
import javax.swing.filechooser.FileNameExtensionFilter;
import javax.swing.plaf.FileChooserUI;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
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
	public static String DIR_IMGS_ENTR;	//Path del directorio de imagens
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
		FileFilter filtroArquitecturaCNN = new FileNameExtensionFilter("Arquitectua de la Red Neuronal (.json)",  "json");
		FileFilter filtroPesosCNN = new FileNameExtensionFilter("Pesos de la Red Neuronal (.bin)", "bin");
		jfc.setAcceptAllFileFilterUsed(false);
		jfc.setFileSelectionMode(JFileChooser.FILES_ONLY);
		jfc.setMultiSelectionEnabled(false);
		
		int opcionElegidaFC = JFileChooser.CANCEL_OPTION;
		int opcionElegidaJP;
		
		String archivoElegido;
		
		try{
			LOG.info("Guardando los pesos de la red neuronal...");
			jfc.setFileFilter(filtroPesosCNN);
			
			while(opcionElegidaFC == JFileChooser.CANCEL_OPTION){
				opcionElegidaFC = jfc.showSaveDialog(null);
				
				if(opcionElegidaFC == JFileChooser.CANCEL_OPTION){
					opcionElegidaJP = JOptionPane.showConfirmDialog(null, 
							"Estas seguro que no quieres guardar los pesos de la red neuronal",
							"",
		    				JOptionPane.YES_NO_OPTION, JOptionPane.QUESTION_MESSAGE);
					
					if(opcionElegidaJP == JOptionPane.YES_OPTION)
						break;
					
				}else{
					archivoElegido = jfc.getSelectedFile().getAbsolutePath();
					if((archivoElegido=validarArchivoAlGuardar(archivoElegido, ".bin"))==null){
						opcionElegidaFC = JFileChooser.CANCEL_OPTION;
						continue;
					}
					
					fos = Files.newOutputStream(Paths.get(archivoElegido));
				    dos = new DataOutputStream(fos);
				    dos.flush();
				    Nd4j.write(model.params(), dos);
				}
			}
		    LOG.info("Guardando los pesos de la red neuronal <ok>");
		    
		    LOG.info("Guardando la arquitectura de la red neuronal...");
		    opcionElegidaFC = JFileChooser.CANCEL_OPTION;
		    jfc.setFileFilter(filtroArquitecturaCNN);
		    
		    while(opcionElegidaFC == JFileChooser.CANCEL_OPTION){
		    	opcionElegidaFC = jfc.showSaveDialog(null);
		    	
		    	if(opcionElegidaFC == JFileChooser.CANCEL_OPTION){
		    		opcionElegidaJP = JOptionPane.showConfirmDialog(null,
		    				"Estas seguro que no quieres guardar la arquitectura de la red neuronal",
		    				"",
		    				JOptionPane.YES_NO_OPTION, JOptionPane.QUESTION_MESSAGE);
		    		
		    		if(opcionElegidaJP == JOptionPane.YES_OPTION)
						break;
		    		
		    	}else{
		    		archivoElegido = jfc.getSelectedFile().getAbsolutePath();
					if((archivoElegido=validarArchivoAlGuardar(archivoElegido, ".json")) == null){
						opcionElegidaFC = JFileChooser.CANCEL_OPTION;
						continue;
					}
						
					
		    		FileUtils.write(new File(archivoElegido),
		    				model.getLayerWiseConfigurations().toJson());
		    	}
		    }
		    LOG.info("Guardando la arquitectura de la red neuronal <ok>");
		    
		}catch(IOException ioe){
			LOG.error("Error al guardar la configuracion de la red neuronal: "+ioe.getMessage());
			throw ioe;
		}catch(Exception e){
			LOG.error("Error inesperado al guardar la red neuronal: "+e.getMessage());
			throw e;
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
	}
	
	
	public static MultiLayerNetwork cargarRedNueronal() throws IOException{
		DataInputStream dis = null;
		
		JFileChooser jfc= new JFileChooser();
		FileFilter filtroArquitecturaCNN = new FileNameExtensionFilter("Arquitectua de la Red Neuronal (.json)",  "json");
		FileFilter filtroPesosCNN = new FileNameExtensionFilter("Pesos de la Red Neuronal (.bin)", "bin");
		
		jfc.setAcceptAllFileFilterUsed(false);
		jfc.setFileSelectionMode(JFileChooser.FILES_ONLY);
		jfc.setMultiSelectionEnabled(false);
		
		try{
			
			LOG.info("Cargando la arquitectura de la red neuronal...");
			MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File("conf.json")));
			MultiLayerNetwork redGuardada = new MultiLayerNetwork(confFromJson);
		    redGuardada.init();
			LOG.info("Cargando la arquitectura de la red neuronal <ok>");
			
			LOG.info("Cargando los pesos de la red neuronal...");
			dis = new DataInputStream(new FileInputStream("coefficients.bin"));
		    INDArray newParams = Nd4j.read(dis);
		    redGuardada.setParameters(newParams);	
			LOG.info("Cargando los pesos de la red neuronal <ok>");
			return redGuardada;
			
		}catch(IOException ioe){
			LOG.error("Error de flujo al cargar la red neuronal: "+ioe.getMessage());
			throw ioe;
		}catch(Exception e){
			LOG.error("Error inesperado al cargar la red neuronal: "+e.getMessage());
			throw e;
		}
		finally{
			try{
				if(dis != null)
					dis.close();				
			}catch(Exception e){
				LOG.error("Error inesperado: ");
				e.printStackTrace();
			}
		}
	}
	
	/**
	 * Funcion que valida que el valor del parametro pathArchivo sea un archivo inexistente
	 * y si existe el archivo le pregunta al usuario si quiere que este se sobreescriba
	 * @param pathArchivo : direccion real del archivo que se quiere guardar
	 * @param extencion : extencion que deberia tener el archivo
	 * @return El path del archivo con la extencion correctamente concatenada
	 */
	public static String validarArchivoAlGuardar(String pathArchivo, String extencion){
		String pathArchivoConExtencion;
		
		if(pathArchivo.endsWith(extencion) && 
				pathArchivo.lastIndexOf('.') == pathArchivo.indexOf('.'))
			pathArchivoConExtencion = pathArchivo;
		else if(!pathArchivo.contains("."))
			pathArchivoConExtencion = pathArchivo+extencion;
		else{
			JOptionPane.showMessageDialog(null, "Nombre incorrecto","",JOptionPane.ERROR_MESSAGE);
			return null;
		}
			
			
		File f = new File(pathArchivoConExtencion);
		int opcionElegidaJP;
		if(f.exists()){
			opcionElegidaJP = JOptionPane.showConfirmDialog(null,
					"El archivo "+f.getName()+" ya existe deseas reemplazarlo",
					"",
    				JOptionPane.YES_NO_OPTION, JOptionPane.QUESTION_MESSAGE);
			if(opcionElegidaJP == JOptionPane.YES_OPTION)
				return pathArchivoConExtencion;
			
			return null;
		}
		return pathArchivoConExtencion;
	}

}
