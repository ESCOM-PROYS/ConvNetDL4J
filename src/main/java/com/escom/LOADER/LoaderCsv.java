package com.escom.LOADER;

import java.io.File;
import java.io.IOException;

import javax.management.InstanceNotFoundException;

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

import com.escom.UTIL.Utileria;

public class LoaderCsv {

	/**
	 * Objeto para imprimir los mensajes de tipo Logger 
	 */
	public static final org.slf4j.Logger LOG = org.slf4j.LoggerFactory.getLogger(LoaderCsv.class);

	/**
	 * Lee las imagenes y las etiquetas de entrenamiento
	 * @return RecordReader :Conjunto ordenado de etiqueta imagen
	 * @throws IOException: Cuando no se puede encontrar los recursos a leer
	 * @throws InterruptedException: Cuando ocurre un error al leer los recursos, un hilo fue interrumpido
	 */
	public static DataSetIterator loadData1() throws IOException, InterruptedException{		
		try {
			LOG.info("Leyendo Imagnes...");
			RecordReader imageReader = new ImageRecordReader(Utileria.WIDTH, Utileria.HEIGHT, Utileria.N_CANALES, false);
			imageReader.initialize(new FileSplit(new File(Utileria.DIR_IMGS_ENTR)));
			LOG.info("Leyendo Imagnes <ok>");
			LOG.info("Leyendo Etiquetas...");
			RecordReader labelsReader = new CSVRecordReader();
			labelsReader.initialize(new FileSplit(new File(Utileria.DIR_LABELS)));
			LOG.info("Leyendo Etiquetas <ok>");

			RecordReader recordReader =  new ComposableRecordReader(imageReader, labelsReader);
			
			DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader, new WritableConverter() {
				@Override
				public Writable convert(Writable writable) throws WritableConverterException {
					try{
						if (writable instanceof Text) {
							String label = writable.toString().replaceAll("\u0000", "");
							int index = Utileria.ETIQUETAS.indexOf(label);
							return new IntWritable(index);
						}else{
							throw new InstanceNotFoundException("ERROR: No es una instancia de Text. [dataSetIterator]");
						}
						
					}catch(InstanceNotFoundException infe){
						LOG.error("Tratando de convertir algo que no es texto: "+infe.getMessage());
						infe.printStackTrace();
					}
					catch(Exception e){
						LOG.error("Error inesperado: "+e.getMessage());
						e.printStackTrace();
					}
					return writable;
				}
			}, Utileria.TAM_LOTE, Utileria.HEIGHT*Utileria.WIDTH*Utileria.N_CANALES, 10);
			
			if(!dataSetIterator.hasNext())
				throw(new InterruptedException("Error: dataSet vacio"));

			// Se sobreescribe convert para sustituir la etiqueta de la imagen por su 
			//indice en la lista de etiquetas:  frog -> 6
			return dataSetIterator;
			
		} catch (IOException ioe) {
			LOG.error("Error al cargar los datos de entrenamiento: "+ioe.getMessage());
			throw ioe;
		} catch (InterruptedException ie) {
			LOG.error("Error interno al cargar con multihilos los datos: "+ie.getMessage());
			throw ie;
		}
	}

}
