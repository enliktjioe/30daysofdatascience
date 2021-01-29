library(shinydashboard)
source("helpers.R")
counties <- readRDS("data/counties.rds")
library(maps)
library(mapproj)

ui <- fluidPage(
  titlePanel("censusVis"),
  
  sidebarLayout(
    sidebarPanel(
      helpText("Create demographic maps with 
               information from the 2010 US Census."),
      
      selectInput("var", 
                  label = "Choose a variable to display",
                  choices = c("Percent White", 
                              "Percent Black",
                              "Percent Hispanic", 
                              "Percent Asian"),
                  selected = "Percent White"),
      
      sliderInput("range", 
                  label = "Range of interest:",
                  min = 0, max = 100, value = c(0, 100))
    ),
    
    mainPanel(plotOutput("map"))
  )
)

server <- function(input, output) {
  
  output$map <- renderPlot({
    # data <- switch (input$var,
    #                 "Percent White" = counties$white,
    #                 "Percent Black" = counties$black,
    #                 "Percent Hispanic" = counties$hispanic,
    #                 "Percent Asian" = counties$asian)
    # 
    # color <- switch(input$var, 
    #                 "Percent White" = "darkgreen",
    #                 "Percent Black" = "black",
    #                 "Percent Hispanic" = "darkorange",
    #                 "Percent Asian" = "darkviolet")
    # 
    # legend <- switch(input$var, 
    #                  "Percent White" = "% White",
    #                  "Percent Black" = "% Black",
    #                  "Percent Hispanic" = "% Hispanic",
    #                  "Percent Asian" = "% Asian")
    # 
    # percent_map(var = data, color, legend, input$range[1], input$range[2])
    
    ## More concise
    
    args <- switch(input$var,
                   "Percent White" = list(counties$white, "darkgreen", "% White"),
                   "Percent Black" = list(counties$black, "black", "% Black"),
                   "Percent Hispanic" = list(counties$hispanic, "darkorange", "% Hispanic"),
                   "Percent Asian" = list(counties$asian, "darkviolet", "% Asian"))
    
    args$min <- input$range[1]
    args$max <- input$range[2]
    
    do.call(percent_map, args)
    
    }
  )
  
}

shinyApp(ui, server)