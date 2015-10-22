package com.escom.UTIL;

import static org.junit.Assert.*;

import org.junit.Test;

import com.escom.LOADER.LoaderCsv;

public class LoaderCsvTest {
	
	/**
	 * Objeto para imprimir los mensajes de tipo Logger 
	 */
	public static final org.slf4j.Logger LOG = org.slf4j.LoggerFactory.getLogger(LoaderCsvTest.class);

	/**
	 * Pruebas para la funcion loadData()
	 */
	@Test
	public void test() {
		LOG.info("Pruebas loadData()");
		try{
			LoaderCsv.loadData();
		}catch(Exception e){
			fail("Error: "+e.getMessage());
		}
	}

}
