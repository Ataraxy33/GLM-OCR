relationship between the beans. The only difference from previous exercises is the change in the JNDI name element tag for the Address home interface:

```xml

<local-jndi-name>AddressHomeLocal</local-jndi-name>

```

Because the Home interface for the Address is local, the tag is `<local-jndi-name>` rather than `<jndi-name>`.

The `weblogic-cmp-rdbms-jar.xml` descriptor file contains a number of new elements and elements in this exercise. A detailed examination of the relationship elements will wait until the next section. The following section will provide a detailed look at the relationship elements.

The file contains a section mapping the `Address` `<camp-field>` attributes from the `ebj-ar.xml` file to the database columns in the `AUDIT` table to form a new section related to the `ebj-ar.xml` file. The `ebj-ar.xml` file contains values in this section.

```xml

<weblogic-rdms-bean>

  <ejb-name>AddressJDB</ejb-name>

  <data-source-name>city</data-source>

  <table-name>ADDRESS</table-name>

  <field-name>

    <cmp-field>id</cmp-field>

    <dbms-column>ID</dbms-column>

  </field-name>

  <field-name>

    <cmp-field>street</cmp-field>

    <dbms-column>STREET</dbms-column>

  </field-name>

  <field-name>

    <cmp-field>city</cmp-field>

    <dbms-column>CITY</dbms-column>

  </field-name>

  <field-name>

    <cmp-field>state</cmp-field>

    <dbms-column>STATE</dbms-column>

  </field-name>

  <field-name>

    <cmp-field>zip</cmp-field>

    <dbms-column>ZIP</dbms-column>

  </field-name>

  <!-- Automatically generate the value of ID in the database on

inserts using sequence table -->

  <automatic-key-generation>

    <generator-type>NAMED SEQUENCE_TABLE</generator-type>

    <generator-name>ADDRESS_SEQUENCE</generator-name>

    <key-cache-size>1</key-cache-size>

  </automatic-key-generation>

</weblogic-rdms-bean>

```