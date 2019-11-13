# data files
set (${PROJECT_NAME}_data
    SKA1_low.txt
    SKA1_mid.txt
    LOFAR_lba.txt
    LOFAR_hba.txt
)

# Copy data files
install(
    FILES
    ${${PROJECT_NAME}_data}
    DESTINATION
    data
)
