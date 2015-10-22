package com.escom.UTIL;

import static org.junit.Assert.*;

import org.junit.Test;

import com.escom.UTIL.Utileria;

public class UtileriaTest {
	/**
	 * Objeto para imprimir los mensajes de tipo Logger 
	 */
	public static final org.slf4j.Logger LOG = org.slf4j.LoggerFactory.getLogger(UtileriaTest.class);

	/**
	 * Pruebas para el metodo getStringPropiedad(String)
	 */
	@Test
	public void testgetStringPropiedad() {
		LOG.info("Test para getStringPropiedad(String)");
		String result;
		try {
			result = Utileria.getStringPropiedad("WIDTH");
			LOG.info("Resultado : "+result);
		} catch (Exception e) {
			e.printStackTrace();
			fail("Error: "+e.getMessage());
		}
		
	}
	
	/**
	 * Pruebas para el metodo getIntPropiedad(String)
	 */
	@Test
	public void testgetIntPropiedad() {
		LOG.info("Test para getIntPropiedad(String)");
		int result;
		try {
			result = Utileria.getIntPropiedad("WIDTH");
			LOG.info("Resultado : "+result);
		} catch (Exception e) {
			e.printStackTrace();
			fail("Error: "+e.getMessage());
		}
		
	}

}
