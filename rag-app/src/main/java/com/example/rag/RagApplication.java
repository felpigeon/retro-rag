package com.example.rag;

import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Main application class for the Retrieval-Augmented Generation (RAG) system.
 * This Spring Boot application serves as the entry point for the RAG service.
 * It integrates with a Python backend service to provide question-answering capabilities.
 */
@SpringBootApplication
@Slf4j
public class RagApplication {

	/**
	 * Main entry point for the application.
	 *
	 * @param args Command-line arguments passed to the application
	 */
	public static void main(String[] args) {
		log.info("Starting RAG Application...");
		SpringApplication.run(RagApplication.class, args);
		log.info("RAG Application started successfully");
	}
}
