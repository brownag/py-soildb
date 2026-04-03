"""
Example of using the metadata parsing functionality.

This example shows how to:
1. Query survey area metadata from sacatalog
2. Parse XML metadata using the metadata module
3. Extract and display useful information
"""

import asyncio

from soildb import Query, SDAClient, parse_survey_metadata


async def main():
    """Demonstrate metadata parsing functionality."""

    # Create client
    client = SDAClient()

    # Get survey area metadata
    print("Fetching survey area metadata...")
    query = Query.from_sql(
        "SELECT areasymbol, areaname, fgdcmetadata FROM sacatalog WHERE areasymbol = 'CA630'"
    )

    response = await client.execute(query)
    data = response.to_pandas()

    if data.empty:
        print("No data found for CA630")
        return

    # Extract the first row
    row = data.iloc[0]
    areasymbol = row["areasymbol"]
    areaname = row["areaname"]
    xml_metadata = row["fgdcmetadata"]

    print(f"\nProcessing metadata for {areasymbol}: {areaname}")

    # Parse the XML metadata
    try:
        metadata = parse_survey_metadata(xml_metadata, areasymbol)

        # Display basic information
        print("\n=== Survey Metadata Summary ===")
        print(f"Area Symbol: {metadata.areasymbol}")
        print(f"Title: {metadata.title}")
        print(f"Publication Date: {metadata.publication_date}")
        if metadata.publication_date_parsed:
            print(
                f"Publication Date (parsed): {metadata.publication_date_parsed.strftime('%B %d, %Y')}"
            )
        print(f"Publisher: {metadata.publisher}")
        print(f"Origin: {metadata.origin}")

        # Display geographic extent
        bbox = metadata.bounding_box
        if any(bbox.values()):
            print("\n=== Geographic Extent ===")
            print(f"West: {bbox['west']:.4f} degrees")
            print(f"East: {bbox['east']:.4f} degrees")
            print(f"North: {bbox['north']:.4f} degrees")
            print(f"South: {bbox['south']:.4f} degrees")

        # Display abstract and purpose
        if metadata.abstract:
            print("\n=== Abstract ===")
            # Truncate long abstracts for display
            abstract = metadata.abstract
            if len(abstract) > 300:
                abstract = abstract[:300] + "..."
            print(abstract)

        if metadata.purpose:
            print("\n=== Purpose ===")
            purpose = metadata.purpose
            if len(purpose) > 300:
                purpose = purpose[:300] + "..."
            print(purpose)

        # Display keywords
        if metadata.keywords:
            print("\n=== Keywords ===")
            print(f"Theme keywords: {', '.join(metadata.theme_keywords)}")
            print(f"Place keywords: {', '.join(metadata.place_keywords)}")

        # Display contact information
        print("\n=== Contact Information ===")
        print(f"Organization: {metadata.contact_organization}")
        print(f"Person: {metadata.contact_person}")
        print(f"Position: {metadata.contact_position}")
        print(f"Email: {metadata.contact_email}")
        print(f"Phone: {metadata.contact_phone}")

        # Display quality information
        print("\n=== Data Quality ===")
        if metadata.attribute_accuracy:
            accuracy = metadata.attribute_accuracy
            if len(accuracy) > 200:
                accuracy = accuracy[:200] + "..."
            print(f"Attribute Accuracy: {accuracy}")

        if metadata.logical_consistency:
            consistency = metadata.logical_consistency
            if len(consistency) > 200:
                consistency = consistency[:200] + "..."
            print(f"Logical Consistency: {consistency}")

        if metadata.completeness:
            completeness = metadata.completeness
            if len(completeness) > 200:
                completeness = completeness[:200] + "..."
            print(f"Completeness: {completeness}")

        # Display processing steps
        steps = metadata.get_process_steps()
        if steps:
            print("\n=== Processing Steps ===")
            for i, step in enumerate(steps, 1):
                print(f"Step {i}:")
                print(f"  Date: {step['date']}")
                if step["description"]:
                    desc = step["description"]
                    if len(desc) > 150:
                        desc = desc[:150] + "..."
                    print(f"  Description: {desc}")
                if step["source_used"]:
                    print(f"  Source: {step['source_used']}")
                print()

        # Display coordinate system information
        print("\n=== Spatial Reference ===")
        print(f"Coordinate System: {metadata.coordinate_system}")
        print(f"Datum: {metadata.datum}")
        print(f"Ellipsoid: {metadata.ellipsoid}")

        # Convert to dictionary for programmatic use
        metadata_dict = metadata.to_dict()
        print("\n=== Dictionary Export ===")
        print(f"Metadata exported as dictionary with {len(metadata_dict)} fields")

        # Display access and use constraints
        if metadata.access_constraints:
            print("\n=== Access Constraints ===")
            constraints = metadata.access_constraints
            if len(constraints) > 300:
                constraints = constraints[:300] + "..."
            print(constraints)

        if metadata.use_constraints:
            print("\n=== Use Constraints ===")
            constraints = metadata.use_constraints
            if len(constraints) > 300:
                constraints = constraints[:300] + "..."
            print(constraints)

    except Exception as e:
        print(f"Error parsing metadata: {e}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
